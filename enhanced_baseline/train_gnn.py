import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Batch

from data_utils import (
    GraphTextDataset,
    collate_graph_text,
    load_id2emb,
    load_descriptions_from_graphs,
)


# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data_baseline" / "data"

TRAIN_GRAPHS = DATA_DIR / "train_graphs.pkl"
VAL_GRAPHS = DATA_DIR / "validation_graphs.pkl"

TRAIN_EMB_CSV = DATA_DIR / "train_embeddings.csv"
VAL_EMB_CSV = DATA_DIR / "validation_embeddings.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-4  # Increased learning rate
WEIGHT_DECAY = 1e-4
HIDDEN = 256
LAYERS = 4
TEMP = 0.1  # Increased temperature (was 0.07, too low can cause numerical issues)
GRAD_CLIP = 1.0  #to prevent exploding gradients

# Enhanced features
FINE_TUNE_BERT = True  #to fine-tune BERT embeddings
BERT_LR = 2e-5  #increased BERT learning rate (was 1e-5)
USE_ATTENTION_READOUT = True  #attention-based readout instead of sum pooling
USE_MULTISCALE = True 
MAX_TEXT_LENGTH = 128  # Max token length for BERT


# ============================================================
# Model definition
# ============================================================
class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionReadout(nn.Module):
    """Attention-based graph readout mechanism."""
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Learnable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
        Returns:
            Graph embeddings [num_graphs, hidden_dim]
        """
        num_graphs = batch.max().item() + 1
        
        k = self.key_proj(x) 
        v = self.value_proj(x)  
        
        #  multi-head attention
        k = k.view(-1, self.num_heads, self.head_dim) 
        v = v.view(-1, self.num_heads, self.head_dim)
        
        q = self.query.expand(num_graphs, -1, -1) 
        q = q.view(num_graphs, self.num_heads, self.head_dim)  
        
        # attention scores per graph
        graph_embs = []
        for g_idx in range(num_graphs):
            mask = (batch == g_idx) 
            if mask.sum() == 0:
                graph_embs.append(torch.zeros(1, self.hidden_dim, device=x.device))
                continue
                
            k_g = k[mask]
            v_g = v[mask]
            q_g = q[g_idx:g_idx+1]
            
            scores = torch.einsum('nhd,mhd->nmh', q_g, k_g) / math.sqrt(self.head_dim) 
            attn_weights = F.softmax(scores, dim=1)  
            
            out = torch.einsum('nmh,mhd->nhd', attn_weights, v_g) 
            out = out.view(1, self.hidden_dim) 
            graph_embs.append(out)
        
        graph_emb = torch.cat(graph_embs, dim=0) 
        graph_emb = self.out_proj(graph_emb)
        graph_emb = self.norm(graph_emb)
        return graph_emb


class FineTunableBERT(nn.Module):
    """Fine-tunable BERT encoder for text embeddings."""
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 128, 
                 fine_tune: bool = True, output_dim: Optional[int] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.fine_tune = fine_tune
        
        # Freeze BERT if not fine-tuning
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_dim = self.bert.config.hidden_size
        if output_dim is not None and output_dim != bert_dim:
            self.proj = nn.Sequential(
                nn.Linear(bert_dim, bert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bert_dim, output_dim),
            )
        else:
            self.proj = nn.Identity()
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: List of text descriptions
        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT embeddings
        outputs = self.bert(**encoded)
        
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        
        # Project if needed
        emb = self.proj(cls_emb)  
        return emb


class MultiFeatureGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int, layers: int, out_dim: int,
                 use_attention_readout: bool = True, use_multiscale: bool = True):
        super().__init__()
        self.use_attention_readout = use_attention_readout
        self.use_multiscale = use_multiscale
        
        self.node_encoder = FeatureProjector(node_dim, hidden)
        self.edge_encoder = FeatureProjector(edge_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp))
            self.norms.append(nn.LayerNorm(hidden))

        # Multi-scale features: store intermediate layer outputs
        if use_multiscale:
            self.multiscale_projs = nn.ModuleList([
                nn.Linear(hidden, hidden) for _ in range(layers)
            ])
        
        # Readout mechanism
        if use_attention_readout:
            self.readout = AttentionReadout(hidden, num_heads=4)
        else:
            self.readout = None  # will use global_add_pool sinon
        

        proj_input_dim = hidden * (1 + layers) if use_multiscale else hidden
        self.proj_head = nn.Sequential(
            nn.Linear(proj_input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        x = self.node_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        multiscale_features = []
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = norm(x + residual)
            
            # Store multi-scale features
            if self.use_multiscale:
                multiscale_features.append(self.multiscale_projs[i](x))

        if self.use_attention_readout:
            graph_emb = self.readout(x, batch.batch)
        else:
            graph_emb = global_add_pool(x, batch.batch)
        
        if self.use_multiscale and multiscale_features:
            scale_embs = [global_add_pool(feat, batch.batch) for feat in multiscale_features]
            graph_emb = torch.cat([graph_emb] + scale_embs, dim=-1)
        
        proj = self.proj_head(graph_emb)
        return F.normalize(proj, dim=-1)


### Training helpers
def info_nce_loss(mol_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    mol_emb = F.normalize(mol_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    logits = mol_emb @ txt_emb.t()
    logits = logits / temperature

    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


@torch.no_grad()
def evaluate(mol_enc: MultiFeatureGNN, bert_encoder: Optional[FineTunableBERT],
             loader: DataLoader, text_dict: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    mol_enc.eval()
    if bert_encoder is not None:
        bert_encoder.eval()
    
    all_mol, all_txt = [], []

    for batch_data in loader:
        if bert_encoder is not None and text_dict is not None:
            graphs, ids = batch_data
            graphs = graphs.to(DEVICE)
            texts = [text_dict[str(id)] for id in ids]
            text_emb = bert_encoder(texts)
        else:
            graphs, text_emb = batch_data
            graphs = graphs.to(DEVICE)
            text_emb = text_emb.to(DEVICE)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        all_mol.append(mol_vec)
        all_txt.append(txt_vec)

    mol_mat = torch.cat(all_mol, dim=0)
    txt_mat = torch.cat(all_txt, dim=0)

    sims = txt_mat @ mol_mat.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = sims.size(0)
    targets = torch.arange(N, device=sims.device)
    pos = (ranks == targets.unsqueeze(1)).nonzero()[:, 1] + 1

    metrics: Dict[str, float] = {
        "MRR": (1.0 / pos.float()).mean().item()
    }
    for k in (1, 5, 10):
        metrics[f"R@{k}"] = (pos <= k).float().mean().item()
    return metrics


def train_epoch(model: MultiFeatureGNN, bert_encoder: Optional[FineTunableBERT], 
                loader: DataLoader, optimizer: torch.optim.Optimizer,
                text_dict: Optional[Dict[str, str]] = None) -> float:
    model.train()
    if bert_encoder is not None:
        bert_encoder.train()
    
    total_loss, total = 0.0, 0

    for batch_data in tqdm(loader, desc="Train", leave=False):
        if bert_encoder is not None and text_dict is not None:
            # Fine-tune BERT: need raw texts
            graphs, ids = batch_data
            graphs = graphs.to(DEVICE)
            texts = [text_dict[str(id)] for id in ids]
            text_emb = bert_encoder(texts)
        else:
            # Use pre-computed embeddings
            graphs, text_emb = batch_data
            graphs = graphs.to(DEVICE)
            text_emb = text_emb.to(DEVICE)

        mol_vec = model(graphs)
        loss = info_nce_loss(mol_vec, text_emb, TEMP)

        optimizer.zero_grad()
        loss.backward()
        
        gnn_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        bert_grad_norm = 0.0
        if bert_encoder is not None:
            bert_grad_norm = torch.nn.utils.clip_grad_norm_(bert_encoder.parameters(), GRAD_CLIP)
        
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    avg_loss = total_loss / max(total, 1)
    
    # debugging
    if total % (BATCH_SIZE * 10) == 0:  # Every 10 batches
        print(f"  [Debug] GNN grad norm: {gnn_grad_norm:.4f}, BERT grad norm: {bert_grad_norm:.4f}")
    
    return avg_loss


def main():
    print(f"Device: {DEVICE}")
    print(f"Fine-tune BERT: {FINE_TUNE_BERT}")
    print(f"Attention readout: {USE_ATTENTION_READOUT}")
    print(f"Multi-scale features: {USE_MULTISCALE}")

    if not TRAIN_GRAPHS.exists():
        raise FileNotFoundError(f"Missing train graphs at {TRAIN_GRAPHS}")

    train_emb = load_id2emb(TRAIN_EMB_CSV) if not FINE_TUNE_BERT else None
    val_emb = load_id2emb(VAL_EMB_CSV) if VAL_EMB_CSV.exists() and not FINE_TUNE_BERT else None
    
    # Load text descriptions if fine-tuning BERT
    train_text_dict = None
    val_text_dict = None
    if FINE_TUNE_BERT:
        train_text_dict = load_descriptions_from_graphs(TRAIN_GRAPHS)
        if VAL_GRAPHS.exists():
            val_text_dict = load_descriptions_from_graphs(VAL_GRAPHS)

    # Create datasets
    if FINE_TUNE_BERT:
        train_ds = GraphTextDataset(TRAIN_GRAPHS, None)
        val_ds = GraphTextDataset(VAL_GRAPHS, None) if VAL_GRAPHS.exists() else None
    else:
        train_ds = GraphTextDataset(TRAIN_GRAPHS, train_emb)
        val_ds = GraphTextDataset(VAL_GRAPHS, val_emb) if val_emb is not None and VAL_GRAPHS.exists() else None
    
    # Custom collate function for fine-tuning
    def collate_fn(batch):
        if FINE_TUNE_BERT:
            graphs = [item if isinstance(item, tuple) else item for item in batch]
            graphs = [g[0] if isinstance(g, tuple) else g for g in graphs]
            batched_graph = Batch.from_data_list(graphs)
            ids = [train_ds.ids[i] for i in range(len(batch))]
            return batched_graph, ids
        else:
            return collate_graph_text(batch)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

    sample = train_ds[0]
    sample_graph = sample[0] if isinstance(sample, tuple) else sample
    node_dim = sample_graph.x.size(-1)
    edge_dim = sample_graph.edge_attr.size(-1)
    
    if FINE_TUNE_BERT:
        emb_dim = 768  # bert-base-uncased hidden size
    else:
        emb_dim = len(next(iter(train_emb.values())))

    model = MultiFeatureGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden=HIDDEN,
        layers=LAYERS,
        out_dim=emb_dim,
        use_attention_readout=USE_ATTENTION_READOUT,
        use_multiscale=USE_MULTISCALE,
    ).to(DEVICE)

    bert_encoder = None
    if FINE_TUNE_BERT:
        bert_encoder = FineTunableBERT(
            model_name='bert-base-uncased',
            max_length=MAX_TEXT_LENGTH,
            fine_tune=True,
            output_dim=emb_dim,
        ).to(DEVICE)
        # Create separate parameter groups with different learning rates
        bert_params = list(bert_encoder.parameters())
        gnn_params = list(model.parameters())
        optimizer = torch.optim.AdamW(
            [
                {'params': bert_params, 'lr': BERT_LR, 'weight_decay': WEIGHT_DECAY},
                {'params': gnn_params, 'lr': LR, 'weight_decay': WEIGHT_DECAY},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, bert_encoder, train_loader, optimizer, train_text_dict)
        log_msg = f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f}"

        if val_loader is not None:
            metrics = evaluate(model, bert_encoder, val_loader, val_text_dict)
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            log_msg += f" | val: {metric_str}"

        print(log_msg)

    ckpt_path = ROOT / "model_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved GNN checkpoint to {ckpt_path}")
    
    if bert_encoder is not None:
        bert_ckpt_path = ROOT / "bert_checkpoint.pt"
        torch.save(bert_encoder.state_dict(), bert_ckpt_path)
        print(f"Saved BERT checkpoint to {bert_ckpt_path}")


if __name__ == "__main__":
    main()

