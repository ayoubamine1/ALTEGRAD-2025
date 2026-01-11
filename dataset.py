"""
Dataset classes for generative molecular captioning.
Loads graphs and tokenizes descriptions for seq2seq training.
"""
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer


class MolecularCaptionDataset(Dataset):
    """
    Dataset for Graph-to-Text generation.
    
    Each item returns:
        - graph: PyG Data object with node/edge features
        - input_ids: Tokenized description (for teacher forcing)
        - attention_mask: Attention mask for tokens
        - labels: Target token IDs (shifted for loss computation)
    """
    
    def __init__(
        self,
        graph_path: str,
        tokenizer_name: str = "laituan245/molt5-small",
        max_length: int = 256,
        is_test: bool = False
    ):
        """
        Args:
            graph_path: Path to .pkl file with list of PyG Data objects
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Max sequence length for tokenization
            is_test: If True, skip tokenization (no descriptions)
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs: List[Data] = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs")
        
        self.is_test = is_test
        self.max_length = max_length
        
        if not is_test:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Pre-tokenize all descriptions for efficiency
            self._tokenize_descriptions()
        else:
            self.tokenizer = None
            self.tokenized = None
    
    def _tokenize_descriptions(self):
        """Pre-tokenize all descriptions."""
        descriptions = [g.description for g in self.graphs]
        
        print(f"Tokenizing {len(descriptions)} descriptions...")
        self.tokenized = self.tokenizer(
            descriptions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        print("Tokenization complete")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        
        if self.is_test:
            return graph
        
        input_ids = self.tokenized["input_ids"][idx]
        attention_mask = self.tokenized["attention_mask"][idx]
        
        # For T5, labels are the same as input_ids
        # The model handles shifting internally
        labels = input_ids.clone()
        # Set padding tokens to -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return graph, input_ids, attention_mask, labels


def collate_fn_generative(batch):
    """
    Collate function for DataLoader.
    
    Handles both training (graph + tokens) and test (graph only) modes.
    """
    if isinstance(batch[0], tuple):
        # Training mode: (graph, input_ids, attention_mask, labels)
        graphs, input_ids, attention_masks, labels = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)
        return batch_graph, input_ids, attention_masks, labels
    else:
        # Test mode: graph only
        return Batch.from_data_list(batch)


def get_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer_name: str = "t5-small",
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0
):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader
    
    train_ds = MolecularCaptionDataset(
        train_path, 
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        is_test=False
    )
    
    val_ds = MolecularCaptionDataset(
        val_path,
        tokenizer_name=tokenizer_name, 
        max_length=max_length,
        is_test=False
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_generative,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_generative,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dl, val_dl, train_ds.tokenizer
