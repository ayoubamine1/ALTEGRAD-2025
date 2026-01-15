# Molecular Graph Captioning
# *Better Retrieval*
The `enhanced_baseline` folder mirrors the baseline pipeline (`data_baseline/`) but swaps in a stronger graph encoder and a contrastive loss. Key upgrades:
- Atom and bond features
- GINE-based encoder
- Contrastive (InfoNCE) loss
- Attention-based graph readout
- Fine-tunable BERT embeddings

`train_gnn.py` : Training entrypoint. Loads data, instantiates the improved encoder, trains with the symmetric InfoNCE loss, evaluates on validation, and saves 
   - `model_checkpoint.pt` - GNN model
   - `bert_checkpoint.pt` - BERT encoder (if fine-tuning enabled)

`retrieval_answer.py` : Uses the trained encoder to embed the test graphs, does nearest-neighbor search against train description embeddings, and writes `test_retrieved_descriptions.csv`. |

# *Generative Approach*

The `app`folder contains a generative model for generating natural language descriptions from molecular graphs. Uses a Graph Neural Network encoder with a T5 decoder, supporting LoRA for efficient fine-tuning.

## Architecture

```
Molecular Graph → EdgeAwareGIN → Q-Former Bridge → T5 Decoder → Text
```

- **EdgeAwareGIN**: 4-layer Graph Isomorphism Network with edge features
- **Q-Former Bridge**: Cross-attention with 32 learnable query tokens  
- **T5 Decoder**: T5-base with optional LoRA fine-tuning

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
pip install peft  # For LoRA support
```

## Quick Start

### Training

```bash
# Standard training (two-phase: frozen → unfrozen LM)
python -m app.train_generative \
    --data_dir data \
    --lm_name t5-base \
    --epochs 20 \
    --batch_size 16

# With LoRA (recommended - more efficient)
python -m app.train_generative \
    --data_dir data \
    --lm_name t5-base \
    --epochs 15 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

### Evaluation

```bash
# Pure generation evaluation
python -m app.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --use_lora

# Hybrid (retrieval + generation) evaluation
python -m app.evaluate_hybrid \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --use_lora \
    --threshold 0.90
```

### Generate Submission

```bash
# Pure generation
python -m app.generate_submission \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --use_lora

# Hybrid (recommended for best BLEU-4)
python -m app.generate_submission_hybrid \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --use_lora \
    --threshold 0.90
```

### RLHF

```bash
# Generate data for DPO
python -m app.rl_data_gen \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --data_dir data

# Train with DPO
python -m app.train_dpo \
    --checkpoint checkpoints/best_model.pt \
    --lm_name t5-base \
    --data_dir data
    --rl_data_csv ./rl_dataset_full.csv
    --lr 1e-6 # Should be low
```

## Project Structure

```
app/
├── model.py              # MolecularCaptioner (main model)
├── graph_encoder.py      # EdgeAwareGIN encoder
├── bridge.py             # Q-Former cross-attention bridge
├── dataset.py            # Data loading and tokenization
├── train_generative.py   # Training script
├── evaluate.py           # Evaluation (BLEU-4, BERTScore)
├── evaluate_hybrid.py    # Hybrid evaluation
├── generate_submission.py        # Pure generation submission
├── generate_submission_hybrid.py # Hybrid submission
├── train_dpo.py          # post training with DPO
└── rl_data_gen.py        # Generate data for DPO 
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--lm_name` | Language model (t5-small, t5-base) | t5-base |
| `--use_lora` | Enable LoRA fine-tuning | False |
| `--lora_r` | LoRA rank | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--threshold` | Similarity threshold for hybrid | 0.90 |
| `--epochs` | Training epochs | 15 |
| `--batch_size` | Batch size | 16 |

## Results

| Method | BLEU-4 | BERTScore | Leaderboard |
|--------|--------|-----------|-------------|
| Pure Generation | 0.247 | 0.969 | 0.55 |
| Hybrid (τ=0.95) | 0.474 | 0.978 | 0.645 |

## Data Format

Place data in `data/` folder:
- `train_graphs.pkl`: List of PyG Data objects with `.description`
- `validation_graphs.pkl`: Validation set
- `test_graphs.pkl`: Test set (no descriptions)

Each graph has:
- `x`: Node features [N, 9] (atom properties)
- `edge_index`: Edge connections [2, E]
- `edge_attr`: Edge features [E, 3] (bond properties)
- `description`: Ground truth text (train/val only)
- `id`: Molecule identifier
