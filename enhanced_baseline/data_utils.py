"""
Improved data utilities for the molecule-text retrieval pipeline.

They extend the baseline helpers by ensuring node/edge tensors exist and are cast
to float32 up-front so the GNN can leverage atom/bond features directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


TensorablePath = Union[str, Path]


def load_id2emb(csv_path: TensorablePath) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed text embeddings from CSV into a dictionary.

    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb: Dict[str, torch.Tensor] = {}
    for _, row in df.iterrows():
        mol_id = str(row["ID"])
        emb_vals = [float(x) for x in str(row["embedding"]).split(",")]
        id2emb[mol_id] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


def load_descriptions_from_graphs(graph_path: TensorablePath) -> Dict[str, str]:
    """Return a mapping of molecule ID -> original text description."""
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)

    id2desc: Dict[str, str] = {}
    for graph in graphs:
        id2desc[str(graph.id)] = graph.description
    return id2desc


class GraphTextDataset(Dataset):
    """
    Dataset that returns PyG graphs (with ensured float features) optionally
    paired with their text embeddings.
    """

    def __init__(
        self,
        graph_path: TensorablePath,
        emb_dict: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.graph_path = Path(graph_path)
        with open(self.graph_path, "rb") as f:
            self.graphs = pickle.load(f)

        self.emb_dict = emb_dict
        self.ids: List[str] = [str(g.id) for g in self.graphs]

        if self.emb_dict is not None:
            missing: List[str] = [g_id for g_id in self.ids if g_id not in self.emb_dict]
            if missing:
                raise KeyError(
                    f"Found {len(missing)} graph IDs without embeddings. "
                    f"Sample missing IDs: {missing[:5]}"
                )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self._ensure_tensor_features(self.graphs[idx])
        if self.emb_dict is None:
            return graph
        text_emb = self.emb_dict[self.ids[idx]]
        return graph, text_emb

    @staticmethod
    def _ensure_tensor_features(graph):
        """
        Guarantee float node features and edge features for downstream GNN layers.
        """
        if not hasattr(graph, "x") or graph.x is None:
            raise ValueError(
                "Graph is missing node features (`x`). "
                "Please recompute preprocessing to include atom features."
            )
        graph.x = graph.x.to(torch.float32)

        # Some splits might not have edge_attr; default to a scalar zero feature.
        if getattr(graph, "edge_attr", None) is None:
            num_edges = graph.edge_index.size(1)
            graph.edge_attr = torch.zeros((num_edges, 1), dtype=torch.float32)
        else:
            graph.edge_attr = graph.edge_attr.to(torch.float32)

        return graph


def collate_graph_text(
    batch: List[Union[Tuple[Batch, torch.Tensor], Batch]]
):
    """
    Collate either:
    - list[(graph, text_emb)] -> (Batch, stacked_text_embs)
    - list[graph] -> Batch
    """
    first = batch[0]
    if isinstance(first, tuple):
        graphs, text_embs = zip(*batch)  # type: ignore[arg-type]
        batched_graph = Batch.from_data_list(list(graphs))
        stacked_embs = torch.stack(text_embs, dim=0)
        return batched_graph, stacked_embs

    return Batch.from_data_list(list(batch))


__all__ = [
    "GraphTextDataset",
    "collate_graph_text",
    "load_id2emb",
    "load_descriptions_from_graphs",
]

