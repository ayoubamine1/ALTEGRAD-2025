"""
Training script for the generative molecular captioner.
Uses teacher forcing with cross-entropy loss.
"""
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from app.dataset import MolecularCaptionDataset, collate_fn_generative
from app.model import create_model, MolecularCaptioner


# =========================================================
# CONFIG
# =========================================================

DEFAULT_CONFIG = {
    "lm_name": "t5-base",
    "batch_size": 16,
    "epochs": 15,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_length": 256,
    "freeze_lm_epochs": 5,  # Freeze LM for first N epochs (ignored if use_lora)
    "gradient_accumulation": 2,
    "save_every": 3,
    "eval_every": 1,
    "use_lora": False,
    "lora_r": 16,
    "lora_alpha": 32,
}


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: MolecularCaptioner,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, (graphs, input_ids, attention_mask, labels) in enumerate(pbar):
        # Move to device
        graphs = graphs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(graphs, labels=labels)
        loss = outputs.loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: MolecularCaptioner,
    dataloader: DataLoader,
    device: torch.device
):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for graphs, input_ids, attention_mask, labels in tqdm(dataloader, desc="Validating"):
        graphs = graphs.to(device)
        labels = labels.to(device)
        
        outputs = model(graphs, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            print(f"Warning: Could not load optimizer state (param count mismatch). Starting fresh optimizer.")
            print(f"  This is normal when resuming after LM unfreezing.")
    return checkpoint["epoch"], checkpoint["loss"]


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    # Build data paths
    train_path = f"{args.data_dir}/train_graphs.pkl"
    val_path = f"{args.data_dir}/validation_graphs.pkl"
    
    train_ds = MolecularCaptionDataset(
        train_path,
        tokenizer_name=args.lm_name,
        max_length=args.max_length,
        is_test=False
    )
    
    val_ds = MolecularCaptionDataset(
        val_path,
        tokenizer_name=args.lm_name,
        max_length=args.max_length,
        is_test=False
    )
    
    # Subset for quick testing
    if args.subset:
        train_ds.graphs = train_ds.graphs[:args.subset]
        train_ds.tokenized = {
            k: v[:args.subset] for k, v in train_ds.tokenized.items()
        }
        val_ds.graphs = val_ds.graphs[:min(100, args.subset)]
        val_ds.tokenized = {
            k: v[:min(100, args.subset)] for k, v in val_ds.tokenized.items()
        }
        print(f"Using subset: {len(train_ds)} train, {len(val_ds)} val")
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_generative,
        num_workers=0,
        pin_memory=True
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_generative,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = create_model(
        lm_name=args.lm_name,
        freeze_lm=not args.use_lora,  # Don't freeze if using LoRA
        use_simple_bridge=args.simple_bridge,
        gradient_checkpointing=True,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    model = model.to(device)
    
    # Optimizer (only for trainable params initially)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_dl) * args.epochs // args.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Unfreeze LM after warmup epochs (only if not using LoRA)
        if not args.use_lora and epoch == args.freeze_lm_epochs and args.freeze_lm_epochs > 0:
            print("Unfreezing LM decoder layers...")
            model.unfreeze_lm(unfreeze_layers=4)  # Last 4 layers
            
            # Recreate optimizer with all parameters
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=args.weight_decay
            )
            trainable, total = model.count_parameters()
            print(f"Trainable parameters now: {trainable:,}")
        
        # Train
        train_loss = train_epoch(
            model, train_dl, optimizer, device,
            accumulation_steps=args.gradient_accumulation
        )
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            val_loss = validate(model, val_dl, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    "checkpoints/best_model.pt"
                )
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                f"checkpoints/epoch_{epoch+1}.pt"
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, train_loss,
        "checkpoints/final_model.pt"
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train molecular captioner")
    
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--lm_name", type=str, default=DEFAULT_CONFIG["lm_name"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--freeze_lm_epochs", type=int, default=DEFAULT_CONFIG["freeze_lm_epochs"])
    parser.add_argument("--gradient_accumulation", type=int, default=DEFAULT_CONFIG["gradient_accumulation"])
    parser.add_argument("--save_every", type=int, default=DEFAULT_CONFIG["save_every"])
    parser.add_argument("--eval_every", type=int, default=DEFAULT_CONFIG["eval_every"])
    parser.add_argument("--subset", type=int, default=None, help="Use subset for testing")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--simple_bridge", action="store_true", help="Use simple bridge")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"], help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"], help="LoRA alpha")
    
    args = parser.parse_args()
    main(args)
