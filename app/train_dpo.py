"""
DPO Training script for Molecular Captioning.
Refines a trained model using Direct Preference Optimization.
Data: Pairs of (Graph, Ground Truth [Chosen], Model Generation [Rejected]).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch

from app.model import create_model

# ==========================================
# DPO Configuration
# ==========================================
DPO_CONFIG = {
    "beta": 0.22,          # The KL penalty coefficient
    "lr": 1e-6,           # DPO requires very low LR
    "batch_size": 128,     # Adjust based on VRAM (A100 80G can handle more)
    "epochs": 1,          # Usually 1-3 epochs is enough
    "gradient_accumulation": 2,
}

class DPODataset(Dataset):
    def __init__(self, graphs_path, csv_path, tokenizer, max_length=512):
        print(f"Loading graphs from {graphs_path}...")
        with open(graphs_path, 'rb') as f:
            self.graph_list = pickle.load(f)
        
        # Create a mapping from ID to Graph object for O(1) access
        self.graph_map = {str(g.id): g for g in self.graph_list}
        
        print(f"Loading pairs from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        # Filter only rows where we have the graph
        self.df['id'] = self.df['id'].astype(str)
        initial_len = len(self.df)
        self.df = self.df[self.df['id'].isin(self.graph_map.keys())]
        
        # Filter out cases where generated == ground_truth (no signal for DPO)
        # self.df = self.df[self.df['generated'] != self.df['ground_truth']]
        
        print(f"Loaded {len(self.df)} pairs (filtered from {initial_len})")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_id = str(row['id'])
        graph = self.graph_map[mol_id]
        
        # Tokenize Chosen (Ground Truth)
        chosen_txt = str(row['ground_truth'])
        chosen_enc = self.tokenizer(
            chosen_txt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        
        # Tokenize Rejected (Model Generation)
        rejected_txt = str(row['generated'])
        rejected_enc = self.tokenizer(
            rejected_txt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        
        return {
            "graph": graph,
            "chosen_input_ids": chosen_enc.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_enc.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_enc.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_enc.attention_mask.squeeze(0),
        }

def collate_dpo(batch):
    graphs = [item['graph'] for item in batch]
    graph_batch = Batch.from_data_list(graphs)
    
    return (
        graph_batch,
        torch.stack([item['chosen_input_ids'] for item in batch]),
        torch.stack([item['chosen_attention_mask'] for item in batch]),
        torch.stack([item['rejected_input_ids'] for item in batch]),
        torch.stack([item['rejected_attention_mask'] for item in batch]),
    )

def get_batch_logps(logits, labels, pad_token_id):
    """
    Compute log probabilities of the labels under the given logits.
    """
    # Shift so that tokens < n predict n
    # T5 / Seq2Seq usually computes loss on labels. 
    # The model forward() returns logits for the labels passed.
    # shape: [B, SeqLen, Vocab]
    
    labels = labels.clone()
    loss_mask = labels != pad_token_id
    
    # Apply log_softmax
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # Mask out padding tokens
    return (per_token_logps * loss_mask).sum(-1)

def train_dpo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Use Training data for DPO
    train_ds = DPODataset(
        f"{args.data_dir}/train_graphs.pkl", 
        args.rl_data_csv, 
        tokenizer,
        max_length=args.max_length
    )
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_dpo,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Load Policy Model (The one we trained)
    print("Loading Policy Model...")
    policy_model = create_model(
        lm_name=args.lm_name,
        freeze_lm=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.to(device)
    policy_model.train()
    
    # 3. Load Reference Model (Copy of Policy, Frozen)
    print("Loading Reference Model...")
    ref_model = create_model(
        lm_name=args.lm_name,
        freeze_lm=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.to(device)
    ref_model.eval()
    # Freeze ref model completely
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # Optimizer
    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 4. Training Loop
    print("\nStarting DPO Training...")
    
    beta = args.beta
    pad_token_id = tokenizer.pad_token_id
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for step, (graphs, chosen_ids, chosen_mask, rejected_ids, rejected_mask) in enumerate(pbar):
            # Move to device
            graphs = graphs.to(device)
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            
            # Replace padding with -100 for T5 labels (standard practice)
            chosen_labels = chosen_ids.clone()
            chosen_labels[chosen_labels == pad_token_id] = -100
            
            rejected_labels = rejected_ids.clone()
            rejected_labels[rejected_labels == pad_token_id] = -100
            
            # ==============================
            # Forward Pass: Policy Model
            # ==============================
            # Get logits for Chosen
            policy_chosen_logits = policy_model(graphs, labels=chosen_labels).logits
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_ids, pad_token_id)
            
            # Get logits for Rejected
            policy_rejected_logits = policy_model(graphs, labels=rejected_labels).logits
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_ids, pad_token_id)
            
            # ==============================
            # Forward Pass: Reference Model
            # ==============================
            with torch.no_grad():
                ref_chosen_logits = ref_model(graphs, labels=chosen_labels).logits
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_ids, pad_token_id)
                
                ref_rejected_logits = ref_model(graphs, labels=rejected_labels).logits
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_ids, pad_token_id)
            
            # ==============================
            # DPO Loss Calculation
            # ==============================
            # pi_logratios = policy_chosen_logps - policy_rejected_logps
            # ref_logratios = ref_chosen_logps - ref_rejected_logps
            # logits = pi_logratios - ref_logratios
            
            # Eq: -log(sigmoid(beta * (log(pi_w/ref_w) - log(pi_l/ref_l))))
            
            policy_log_ratios = policy_chosen_logps - policy_rejected_logps
            ref_log_ratios = ref_chosen_logps - ref_rejected_logps
            
            losses = -F.logsigmoid(beta * (policy_log_ratios - ref_log_ratios))
            loss = losses.mean() / args.gradient_accumulation
            
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation
            
            # Logging
            reward_chosen = (policy_chosen_logps - ref_chosen_logps).detach().mean().item()
            reward_rejected = (policy_rejected_logps - ref_rejected_logps).detach().mean().item()
            reward_acc = ((policy_log_ratios - ref_log_ratios) > 0).float().mean().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation:.4f}",
                "acc": f"{reward_acc:.2f}",
                "margin": f"{reward_chosen - reward_rejected:.3f}"
            })
            
    # Save Model
    os.makedirs("checkpoints_dpo", exist_ok=True)
    save_path = "checkpoints_dpo/dpo_final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": policy_model.state_dict(),
        "loss": total_loss / len(train_dl)
    }, save_path)
    print(f"\nDPO training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--rl_data_csv", type=str, required=True, help="Generated RL dataset CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint")
    parser.add_argument("--lm_name", type=str, default="laituan245/molt5-large")
    parser.add_argument("--batch_size", type=int, default=DPO_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DPO_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DPO_CONFIG["lr"])
    parser.add_argument("--beta", type=float, default=DPO_CONFIG["beta"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation", type=int, default=DPO_CONFIG["gradient_accumulation"])
    
    # LoRA config (must match base model)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    train_dpo(args)