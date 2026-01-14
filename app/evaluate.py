"""
Evaluation script for molecular captioner.
Computes BLEU-4 and BERTScore on validation set.
"""
import argparse
from typing import List, Dict
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# BLEU
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# BERTScore (optional - install with: pip install bert-score)
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert-score not installed. Run: pip install bert-score")

from app.dataset import MolecularCaptionDataset, collate_fn_generative
from app.model import create_model


def compute_bleu4(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-4 score.
    
    Args:
        predictions: List of generated captions
        references: List of ground truth captions
        
    Returns:
        Dictionary with BLEU scores
    """
    # Tokenize for BLEU
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]  # BLEU expects list of lists
    
    smoother = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    
    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
    }


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BERTScore using RoBERTa-base.
    
    Args:
        predictions: List of generated captions
        references: List of ground truth captions
        
    Returns:
        Dictionary with BERTScore metrics
    """
    if not HAS_BERTSCORE:
        return {"BERTScore-F1": -1.0, "BERTScore-P": -1.0, "BERTScore-R": -1.0}
    
    P, R, F1 = bert_score(
        predictions, 
        references, 
        model_type="seyonec/ChemBERTa-zinc-base-v1",
        num_layers=6,  # ChemBERTa has 12 layers
        verbose=False
    )
    
    return {
        "BERTScore-P": P.mean().item(),
        "BERTScore-R": R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }


def generate_captions(model, dataloader, tokenizer, device, max_length=256, num_beams=5, repetition_penalty=1.5, no_repeat_ngram_size=3):
    """
    Generate captions for all samples in dataloader.
    
    Returns:
        predictions: List of generated captions
        references: List of ground truth captions
        ids: List of molecule IDs
    """
    model.eval()
    predictions = []
    references = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            graphs, input_ids, attention_mask, labels = batch
            graphs = graphs.to(device)
            
            # Generate
            generated_ids = model.generate(
                graphs,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            
            # Decode
            generated_texts = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            predictions.extend(generated_texts)
            
            # Get reference texts (decode labels)
            # Replace -100 with pad token for decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(
                labels_for_decode,
                skip_special_tokens=True
            )
            references.extend(ref_texts)
            
            # Get IDs
            for g in graphs.to_data_list():
                ids.append(g.id)
    
    return predictions, references, ids


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    # Load dataset
    print(f"Loading {args.split} set...")
    
    if args.split == "validation":
        graph_path = f"{args.data_dir}/validation_graphs.pkl"
    else:
        graph_path = f"{args.data_dir}/{args.split}_graphs.pkl"
    
    dataset = MolecularCaptionDataset(
        graph_path,
        tokenizer_name=args.lm_name,
        max_length=args.max_length,
        is_test=False
    )
    
    if args.subset:
        dataset.graphs = dataset.graphs[:args.subset]
        dataset.tokenized = {k: v[:args.subset] for k, v in dataset.tokenized.items()}
        print(f"Using subset of {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_generative,
        num_workers=0
    )
    
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
    print("\nGenerating captions...")
    predictions, references, ids = generate_captions(
        model, dataloader, tokenizer, device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    
    bleu_scores = compute_bleu4(predictions, references)
    print("\nBLEU Scores:")
    for k, v in bleu_scores.items():
        print(f"  {k}: {v:.4f}")
    
    bertscore = compute_bertscore(predictions, references)
    print("\nBERTScore:")
    for k, v in bertscore.items():
        print(f"  {k}: {v:.4f}")
    
    # Print sample predictions
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    for i in range(min(5, len(predictions))):
        print(f"\nID: {ids[i]}")
        print(f"Reference: {references[i][:200]}...")
        print(f"Predicted: {predictions[i][:200]}...")
    
    # Save results
    if args.output:
        import pandas as pd
        results_df = pd.DataFrame({
            "ID": ids,
            "reference": references,
            "prediction": predictions
        })
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")
    
    return {**bleu_scores, **bertscore}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate molecular captioner")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    main(args)
