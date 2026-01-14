"""
Hybrid submission generator: combines retrieval + generative approaches.
- Uses retrieval for molecules similar to training data (high BLEU-4)
- Falls back to generative model for novel molecules
"""
import argparse
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from app.model import create_model
from app.graph_encoder import EdgeAwareGIN


def collate_graphs(batch):
    """Collate graphs into batch."""
    return Batch.from_data_list(batch)


def compute_graph_embeddings(graphs, encoder, device, batch_size=32):
    """Compute graph embeddings using trained GNN encoder."""
    embeddings = []
    
    dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            batch = batch.to(device)
            # Get graph-level embeddings (not node-level)
            emb = encoder(batch, return_node_embeddings=False)
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def find_similar_molecules(test_embs, train_embs, top_k=1):
    """Find most similar training molecules for each test molecule."""
    # Normalize for cosine similarity
    test_norm = F.normalize(test_embs, dim=1)
    train_norm = F.normalize(train_embs, dim=1)
    
    # Compute similarity matrix
    similarities = torch.mm(test_norm, train_norm.t())  # [N_test, N_train]
    
    # Get top-k matches
    top_sims, top_indices = similarities.topk(top_k, dim=1)
    
    return top_sims, top_indices


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load training graphs
    print(f"\nLoading training graphs from {args.data_dir}/train_graphs.pkl...")
    with open(f"{args.data_dir}/train_graphs.pkl", 'rb') as f:
        train_graphs = pickle.load(f)
    print(f"Loaded {len(train_graphs)} training graphs")
    
    # Load validation graphs (also has descriptions!)
    print(f"\nLoading validation graphs from {args.data_dir}/validation_graphs.pkl...")
    with open(f"{args.data_dir}/validation_graphs.pkl", 'rb') as f:
        val_graphs = pickle.load(f)
    print(f"Loaded {len(val_graphs)} validation graphs")
    
    # Combine train + validation for retrieval pool
    reference_graphs = train_graphs + val_graphs
    reference_descriptions = [g.description for g in reference_graphs]
    print(f"\nTotal reference pool: {len(reference_graphs)} molecules (train + val)")
    
    # Load test graphs
    print(f"\nLoading test graphs from {args.data_dir}/test_graphs.pkl...")
    with open(f"{args.data_dir}/test_graphs.pkl", 'rb') as f:
        test_graphs = pickle.load(f)
    print(f"Loaded {len(test_graphs)} test graphs")
    
    if args.subset:
        test_graphs = test_graphs[:args.subset]
    
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
    
    # Get GNN encoder from model
    gnn_encoder = model.graph_encoder
    gnn_encoder.eval()
    
    # Compute embeddings for reference pool (train + val)
    print("\nComputing reference graph embeddings (train + val)...")
    reference_embs = compute_graph_embeddings(reference_graphs, gnn_encoder, device, args.batch_size)
    
    print("Computing test graph embeddings...")
    test_embs = compute_graph_embeddings(test_graphs, gnn_encoder, device, args.batch_size)
    
    # Find similar molecules in combined pool
    print("\nFinding similar molecules in reference pool...")
    top_sims, top_indices = find_similar_molecules(test_embs, reference_embs, top_k=1)
    
    # Analyze similarity distribution
    sims = top_sims[:, 0].numpy()
    print(f"Similarity stats: min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}")
    print(f"Molecules above threshold {args.threshold}: {(sims >= args.threshold).sum()} / {len(sims)}")
    
    # Load tokenizer for generative model
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Generate predictions
    print(f"\nGenerating predictions (threshold={args.threshold})...")
    
    all_ids = []
    all_descriptions = []
    retrieval_count = 0
    generation_count = 0
    
    # Process in batches for generation
    gen_dataloader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(gen_dataloader, desc="Generating"):
            batch_size = batch.num_graphs
            batch = batch.to(device)
            
            # Get IDs
            for g in batch.to_data_list():
                all_ids.append(g.id)
            
            # Check similarity for each molecule in batch
            batch_descriptions = []
            need_generation = []
            need_generation_indices = []
            
            for i in range(batch_size):
                similarity = top_sims[idx + i, 0].item()
                
                if similarity >= args.threshold:
                    # Use retrieved description from reference pool
                    ref_idx = top_indices[idx + i, 0].item()
                    batch_descriptions.append(reference_descriptions[ref_idx])
                    retrieval_count += 1
                else:
                    # Mark for generation
                    batch_descriptions.append(None)
                    need_generation.append(i)
                    need_generation_indices.append(idx + i)
                    generation_count += 1
            
            # Generate for molecules that need it
            if need_generation:
                # Create sub-batch for generation
                graphs_to_gen = [test_graphs[idx + i] for i in need_generation]
                gen_batch = Batch.from_data_list(graphs_to_gen).to(device)
                
                generated_ids = model.generate(
                    gen_batch,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size
                )
                
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Fill in generated descriptions
                for i, gen_idx in enumerate(need_generation):
                    batch_descriptions[gen_idx] = generated_texts[i]
            
            all_descriptions.extend(batch_descriptions)
            idx += batch_size
    
    print(f"\nRetrieval: {retrieval_count}, Generation: {generation_count}")
    
    # Create submission
    submission_df = pd.DataFrame({
        "ID": all_ids,
        "description": all_descriptions
    })
    
    submission_df.to_csv(args.output, index=False)
    print(f"\nSaved submission to {args.output}")
    
    # Print samples
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    for i in range(min(5, len(submission_df))):
        sim = top_sims[i, 0].item()
        method = "RETRIEVAL" if sim >= args.threshold else "GENERATED"
        print(f"\nID: {submission_df.iloc[i]['ID']} [{method}, sim={sim:.3f}]")
        desc = submission_df.iloc[i]['description']
        print(f"Description: {desc[:200]}..." if len(desc) > 200 else f"Description: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval+generation submission")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="submission_hybrid.csv")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.90, 
                        help="Similarity threshold for using retrieval (0-1)")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for debugging")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    main(args)
