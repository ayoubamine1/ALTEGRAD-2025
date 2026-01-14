import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_utils import (
    GraphTextDataset,
    collate_graph_text,
    load_descriptions_from_graphs,
    load_id2emb,
)
from train_gnn import (
    MultiFeatureGNN, 
    FineTunableBERT,
    HIDDEN, 
    LAYERS, 
    DATA_DIR,
    FINE_TUNE_BERT,
    USE_ATTENTION_READOUT,
    USE_MULTISCALE,
    MAX_TEXT_LENGTH,
)


ROOT = Path(__file__).resolve().parent
TRAIN_GRAPHS = DATA_DIR / "train_graphs.pkl"
TEST_GRAPHS = DATA_DIR / "test_graphs.pkl"
TRAIN_EMB_CSV = DATA_DIR / "train_embeddings.csv"

CKPT_PATH = ROOT / "model_checkpoint.pt"
BERT_CKPT_PATH = ROOT / "bert_checkpoint.pt"
OUTPUT_CSV = ROOT / "test_retrieved_descriptions_improv5.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}. Run train_gnn.py first.")
    if not TEST_GRAPHS.exists():
        raise FileNotFoundError(f"Missing test graphs at {TEST_GRAPHS}")

    # using fine-tuned BERT?
    use_fine_tuned_bert = BERT_CKPT_PATH.exists()
    print(f"Using fine-tuned BERT: {use_fine_tuned_bert}")

    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    train_ids = list(train_desc.keys())
    
    feat_ds = GraphTextDataset(TRAIN_GRAPHS)
    sample_graph = feat_ds[0]
    node_dim = sample_graph.x.size(-1)
    edge_dim = sample_graph.edge_attr.size(-1)
    
    if use_fine_tuned_bert:
        emb_dim = 768  # BERT base hidden size
    else:
        if not TRAIN_EMB_CSV.exists():
            raise FileNotFoundError(f"Train embeddings not found at {TRAIN_EMB_CSV}. Run generate_description_embeddings.py first.")
        train_emb = load_id2emb(TRAIN_EMB_CSV)
        emb_dim = len(next(iter(train_emb.values())))

    # Load GNN model
    model = MultiFeatureGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden=HIDDEN,
        layers=LAYERS,
        out_dim=emb_dim,
        use_attention_readout=USE_ATTENTION_READOUT,
        use_multiscale=USE_MULTISCALE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    # Load BERT encoder if fine-tuned
    bert_encoder = None
    if use_fine_tuned_bert:
        bert_encoder = FineTunableBERT(
            model_name='bert-base-uncased',
            max_length=MAX_TEXT_LENGTH,
            fine_tune=False,  # Inference mode
            output_dim=emb_dim,
        ).to(DEVICE)
        bert_encoder.load_state_dict(torch.load(BERT_CKPT_PATH, map_location=DEVICE))
        bert_encoder.eval()
        
        # Generate train embeddings using fine-tuned BERT
        print("Generating train embeddings with fine-tuned BERT")
        train_texts = [train_desc[_id] for _id in train_ids]
        train_embs = []
        batch_size = 64
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            batch_embs = bert_encoder(batch_texts)
            train_embs.append(batch_embs)
        train_embs = torch.cat(train_embs, dim=0)
        train_embs = F.normalize(train_embs, dim=-1)
    else:
        # Use pre-computed embeddings
        train_emb = load_id2emb(TRAIN_EMB_CSV)
        train_embs = torch.stack([train_emb[_id] for _id in train_ids]).to(DEVICE)
        train_embs = F.normalize(train_embs, dim=-1)

    test_ds = GraphTextDataset(TEST_GRAPHS)
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_graph_text,
    )

    test_embs = []
    ordered_ids = []
    cursor = 0
    for graphs in test_loader:
        graphs = graphs.to(DEVICE)
        mol_emb = model(graphs)
        test_embs.append(mol_emb)

        count = graphs.num_graphs
        batch_ids = test_ds.ids[cursor:cursor + count]
        ordered_ids.extend(batch_ids)
        cursor += count

    test_embs = torch.cat(test_embs, dim=0)
    sims = test_embs @ train_embs.t()
    best_idx = sims.argmax(dim=-1).cpu()

    rows = []
    for test_id, idx in zip(ordered_ids, best_idx):
        train_id = train_ids[idx]
        rows.append({
            "ID": test_id,
            "description": train_desc[train_id],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} retrieved descriptions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

