"""
Evaluate hybrid retrieval+generation approach on validation set.
This lets us measure BLEU-4 improvement before submitting to leaderboard.
"""
import argparse
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

from app.model import create_model


def collate_graphs(batch):
    return Batch.from_data_list(batch)


def compute_graph_embeddings(graphs, encoder, device, batch_size=32):
    embeddings = []
    dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            batch = batch.to(device)
            emb = encoder(batch, return_node_embeddings=False)
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def compute_bleu4(predictions, references):
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    smoother = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    
    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}


def compute_bertscore(predictions, references):
    if not HAS_BERTSCORE:
        return {"BERTScore-F1": -1.0}
    
    P, R, F1 = bert_score(predictions, references, model_type="seyonec/ChemBERTa-zinc-base-v1", num_layers=6, verbose=False)
    return {"BERTScore-P": P.mean().item(), "BERTScore-R": R.mean().item(), "BERTScore-F1": F1.mean().item()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load graphs
    print(f"\nLoading training graphs...")
    with open(f"{args.data_dir}/train_graphs.pkl", 'rb') as f:
        train_graphs = pickle.load(f)
    train_descriptions = [g.description for g in train_graphs]
    print(f"Loaded {len(train_graphs)} training graphs")
    
    print(f"\nLoading validation graphs...")
    with open(f"{args.data_dir}/validation_graphs.pkl", 'rb') as f:
        val_graphs = pickle.load(f)
    print(f"Loaded {len(val_graphs)} validation graphs")
    
    if args.subset:
        val_graphs = val_graphs[:args.subset]
        print(f"Using subset of {len(val_graphs)} validation samples")
    
    val_descriptions = [g.description for g in val_graphs]
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(
        lm_name=args.lm_name, 
        freeze_lm=False, 
        gradient_checkpointing=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    gnn_encoder = model.graph_encoder
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Compute embeddings
    print("\nComputing embeddings...")
    train_embs = compute_graph_embeddings(train_graphs, gnn_encoder, device, args.batch_size)
    val_embs = compute_graph_embeddings(val_graphs, gnn_encoder, device, args.batch_size)
    
    # Normalize for cosine similarity
    train_norm = F.normalize(train_embs, dim=1)
    val_norm = F.normalize(val_embs, dim=1)
    
    # Find similarities
    similarities = torch.mm(val_norm, train_norm.t())
    top_sims, top_indices = similarities.max(dim=1)
    
    sims = top_sims.numpy()
    print(f"\nSimilarity stats: min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}")
    print(f"Above threshold {args.threshold}: {(sims >= args.threshold).sum()} / {len(sims)}")
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    predictions = []
    methods = []
    
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating"):
            batch_size = batch.num_graphs
            batch = batch.to(device)
            
            batch_preds = []
            need_gen = []
            
            for i in range(batch_size):
                sim = top_sims[idx + i].item()
                if sim >= args.threshold:
                    train_idx = top_indices[idx + i].item()
                    batch_preds.append(train_descriptions[train_idx])
                    methods.append("RETRIEVAL")
                else:
                    batch_preds.append(None)
                    need_gen.append(i)
                    methods.append("GENERATED")
            
            if need_gen:
                graphs_to_gen = [val_graphs[idx + i] for i in need_gen]
                gen_batch = Batch.from_data_list(graphs_to_gen).to(device)
                
                gen_ids = model.generate(
                    gen_batch, max_length=args.max_length, num_beams=args.num_beams,
                    repetition_penalty=1.5, no_repeat_ngram_size=3
                )
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                for i, gen_idx in enumerate(need_gen):
                    batch_preds[gen_idx] = gen_texts[i]
            
            predictions.extend(batch_preds)
            idx += batch_size
    
    # Compute metrics
    print("\n" + "="*60)
    print("HYBRID METHOD RESULTS")
    print("="*60)
    
    retrieval_count = methods.count("RETRIEVAL")
    generation_count = methods.count("GENERATED")
    print(f"\nRetrieval: {retrieval_count} ({100*retrieval_count/len(methods):.1f}%)")
    print(f"Generation: {generation_count} ({100*generation_count/len(methods):.1f}%)")
    
    bleu = compute_bleu4(predictions, val_descriptions)
    print("\nBLEU Scores:")
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")
    
    bert = compute_bertscore(predictions, val_descriptions)
    print("\nBERTScore:")
    for k, v in bert.items():
        print(f"  {k}: {v:.4f}")
    
    # Estimated leaderboard
    composite = 0.5 * bleu["BLEU-4"] + 0.5 * bert.get("BERTScore-F1", 0)
    print(f"\nEstimated Composite: {composite:.4f}")
    
    # Samples
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)
    for i in range(min(5, len(predictions))):
        print(f"\nID: {val_graphs[i].id} [{methods[i]}, sim={top_sims[i]:.3f}]")
        print(f"Reference: {val_descriptions[i][:150]}...")
        print(f"Predicted: {predictions[i][:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--subset", type=int, default=None)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    main(args)
