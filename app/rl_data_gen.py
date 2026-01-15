"""
Script to generate an RL dataset (Inputs, Ground Truths, Model Predictions).
Runs the trained model on both Training and Validation sets to create a baseline
for Reinforcement Learning (e.g., SCST, RLHF).
"""
import os
import argparse
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from app.model import create_model

def collate_graphs(batch):
    """Collate graphs into batch."""
    return Batch.from_data_list(batch)

def run_generation(model, tokenizer, graphs, device, args, dataset_name="Dataset"):
    """
    Run generation on a list of graphs and return a list of dictionaries.
    """
    results = []
    
    dataloader = DataLoader(
        graphs, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True
    )
    
    # Ensure model is in eval mode
    model.eval()
    
    # Use bfloat16 for A100 efficiency
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        for batch in tqdm(dataloader, desc=f"Generating for {dataset_name}"):
            # Move batch to device
            batch = batch.to(device)
            batch_size = batch.num_graphs
            
            # 1. Generate Captions
            # For RL datasets, we typically use beam search to get high quality baselines,
            # or sampling if you want to explore. Here we stick to beam search as a strong baseline.
            generated_ids = model.generate(
                batch,
                max_length=args.max_length,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                # early_stopping=True
            )
            
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 2. Extract Metadata and Ground Truth
            # Convert batch back to list to access individual graph attributes
            data_list = batch.to_data_list()
            
            for i, graph in enumerate(data_list):
                mol_id = getattr(graph, 'id', None)
                # Try to get SMILES if available, useful for context
                smiles = getattr(graph, 'smiles', "") 
                # Ground Truth Description
                ground_truth = getattr(graph, 'description', "")
                
                results.append({
                    "split": dataset_name,
                    "id": mol_id,
                    "smiles": smiles,
                    "ground_truth": ground_truth,
                    "generated": generated_texts[i]
                })
                
    return results

def main(args):
    # Setup Device
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Tokenizer
    print(f"Loading tokenizer: {args.lm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Load Model
    print(f"Loading model architecture: {args.lm_name}")
    model = create_model(
        lm_name=args.lm_name,
        freeze_lm=False, # We need the full model for generation
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Check if we need to fix keys (sometimes 'module.' prefix exists if trained with DataParallel)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_results = []

    # ==========================================
    # 1. Process Training Data
    # ==========================================
    train_path = os.path.join(args.data_dir, "train_graphs.pkl")
    if os.path.exists(train_path):
        print(f"\nLoading training graphs from {train_path}...")
        with open(train_path, 'rb') as f:
            train_graphs = pickle.load(f)
            # train_graphs = train_graphs[:10] #TO DELETE
        
        # Optional: Subset for testing
        if args.subset:
            train_graphs = train_graphs[:args.subset]
            
        print(f"Processing {len(train_graphs)} training graphs...")
        train_results = run_generation(model, tokenizer, train_graphs, device, args, dataset_name="train")
        all_results.extend(train_results)
    else:
        print(f"Warning: {train_path} not found.")

    # ==========================================
    # 2. Process Validation Data
    # ==========================================
    val_path = os.path.join(args.data_dir, "validation_graphs.pkl")
    if os.path.exists(val_path):
        print(f"\nLoading validation graphs from {val_path}...")
        with open(val_path, 'rb') as f:
            val_graphs = pickle.load(f)
            # val_graphs = val_graphs[:10] #TO DELETE
            
        if args.subset:
            val_graphs = val_graphs[:args.subset]

        print(f"Processing {len(val_graphs)} validation graphs...")
        val_results = run_generation(model, tokenizer, val_graphs, device, args, dataset_name="validation")
        all_results.extend(val_results)
    else:
        print(f"Warning: {val_path} not found.")

    # ==========================================
    # 3. Save to CSV
    # ==========================================
    df = pd.DataFrame(all_results)
    
    # Calculate simple exact match accuracy just for sanity check
    exact_matches = (df['ground_truth'] == df['generated']).sum()
    print(f"\nTotal samples processed: {len(df)}")
    print(f"Exact matches (sanity check): {exact_matches / len(df):.4f}")
    
    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"\nRL Dataset saved to: {output_path}")
    print("Columns: ", list(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RL Dataset (GT vs Generated)")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data folder containing .pkl files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="rl_dataset.csv", help="Output CSV filename")
    
    # Model Config
    parser.add_argument("--lm_name", type=str, default="laituan245/molt5-large")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Generation Config (Parameters used to create the 'Baseline')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    
    parser.add_argument("--subset", type=int, default=None, help="Process only N samples for debugging")
    
    args = parser.parse_args()
    main(args)