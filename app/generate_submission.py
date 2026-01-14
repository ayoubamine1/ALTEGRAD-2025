"""
Generate submission file for Kaggle.
Creates CSV with ID and generated descriptions for test set.
"""
import argparse
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from app.dataset import MolecularCaptionDataset
from app.model import create_model


def collate_test(batch):
    """Collate for test set (graphs only)."""
    return Batch.from_data_list(batch)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build test path
    test_path = f"{args.data_dir}/test_graphs.pkl"
    
    # Load test graphs
    print(f"Loading test graphs from {test_path}...")
    with open(test_path, 'rb') as f:
        test_graphs = pickle.load(f)
    print(f"Loaded {len(test_graphs)} test graphs")
    
    # Create dataloader
    dataloader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_test
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
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
    
    # Generate captions
    print("\nGenerating captions for test set...")
    
    all_ids = []
    all_descriptions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            batch = batch.to(device)
            
            # Generate
            generated_ids = model.generate(
                batch,
                max_length=args.max_length,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else 1.0,
                top_p=args.top_p if args.do_sample else 1.0
            )
            
            # Decode
            generated_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Get IDs
            for g in batch.to_data_list():
                all_ids.append(g.id)
            
            all_descriptions.extend(generated_texts)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "ID": all_ids,
        "description": all_descriptions
    })
    
    # Save
    output_path = args.output
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSaved submission to {output_path}")
    print(f"Total samples: {len(submission_df)}")
    
    # Print samples
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    for i in range(min(5, len(submission_df))):
        print(f"\nID: {submission_df.iloc[i]['ID']}")
        desc = submission_df.iloc[i]['description']
        print(f"Description: {desc[:200]}..." if len(desc) > 200 else f"Description: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission CSV")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    main(args)
