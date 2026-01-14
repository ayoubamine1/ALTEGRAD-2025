"""
Feature-aware Graph Neural Network encoder for molecular graphs.
Uses all 9 node features and 3 edge features via embedding lookups.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


# =========================================================
# Feature Vocabulary Sizes (from data_utils.py)
# =========================================================
NODE_FEATURE_DIMS = [
    119,  # atomic_num: 0-118
    9,    # chirality: 9 types
    11,   # degree: 0-10
    12,   # formal_charge: -5 to +6
    9,    # num_hs: 0-8
    5,    # num_radical_electrons: 0-4
    8,    # hybridization: 8 types
    2,    # is_aromatic: bool
    2,    # is_in_ring: bool
]

EDGE_FEATURE_DIMS = [
    22,   # bond_type: 22 types
    6,    # stereo: 6 types
    2,    # is_conjugated: bool
]


class NodeEmbedding(nn.Module):
    """
    Embeds all 9 categorical node features and concatenates them.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim)
            for num_classes in NODE_FEATURE_DIMS
        ])
        self.total_dim = embed_dim * len(NODE_FEATURE_DIMS)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features tensor [N, 9] with categorical indices
        Returns:
            Embedded features [N, embed_dim * 9]
        """
        embedded = []
        for i, embed_layer in enumerate(self.embeddings):
            # Clamp to valid range to avoid index errors
            feat = x[:, i].clamp(0, NODE_FEATURE_DIMS[i] - 1)
            embedded.append(embed_layer(feat))
        return torch.cat(embedded, dim=-1)


class EdgeEmbedding(nn.Module):
    """
    Embeds all 3 categorical edge features and concatenates them.
    """
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim)
            for num_classes in EDGE_FEATURE_DIMS
        ])
        self.total_dim = embed_dim * len(EDGE_FEATURE_DIMS)
        
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: Edge features tensor [E, 3] with categorical indices
        Returns:
            Embedded features [E, embed_dim * 3]
        """
        embedded = []
        for i, embed_layer in enumerate(self.embeddings):
            feat = edge_attr[:, i].clamp(0, EDGE_FEATURE_DIMS[i] - 1)
            embedded.append(embed_layer(feat))
        return torch.cat(embedded, dim=-1)


class EdgeAwareGIN(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE).
    
    Uses all 9 node features and 3 edge features for a rich molecular
    representation. Outputs both node-level and graph-level embeddings.
    """
    
    def __init__(
        self,
        node_embed_dim: int = 64,
        edge_embed_dim: int = 32,
        hidden_dim: int = 256,
        out_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature embeddings
        self.node_embed = NodeEmbedding(node_embed_dim)
        self.edge_embed = EdgeEmbedding(edge_embed_dim)
        
        node_input_dim = self.node_embed.total_dim  # 64 * 9 = 576
        edge_input_dim = self.edge_embed.total_dim  # 32 * 3 = 96
        
        # Project embedded features to hidden dimension
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)
        
        # GINE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # MLP for GIN aggregation
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
    def forward(self, batch: Batch, return_node_embeddings: bool = False):
        """
        Args:
            batch: PyG Batch object with x, edge_index, edge_attr, batch
            return_node_embeddings: If True, return node embeddings instead of graph
            
        Returns:
            If return_node_embeddings:
                node_emb: [N, out_dim] node-level embeddings
            Else:
                graph_emb: [B, out_dim] graph-level embeddings
        """
        # Embed categorical features
        x = self.node_embed(batch.x)  # [N, 576]
        edge_attr = self.edge_embed(batch.edge_attr)  # [E, 96]
        
        # Project to hidden dimension
        h = self.node_proj(x)  # [N, hidden_dim]
        edge_h = self.edge_proj(edge_attr)  # [E, hidden_dim]
        
        # Message passing layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, edge_h)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # Residual connection
        
        # Output projection
        h = self.out_proj(h)  # [N, out_dim]
        
        if return_node_embeddings:
            return h
        
        # Global pooling for graph-level representation
        graph_emb = global_mean_pool(h, batch.batch)  # [B, out_dim]
        return graph_emb
    
    def get_node_embeddings(self, batch: Batch) -> torch.Tensor:
        """Get node-level embeddings for cross-attention in bridge."""
        return self.forward(batch, return_node_embeddings=True)


if __name__ == "__main__":
    # Quick test
    print("Testing EdgeAwareGIN...")
    # Create dummy data
    from torch_geometric.data import Data
    
    x = torch.randint(0, 10, (5, 9))  # 5 nodes, 9 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # 4 edges
    edge_attr = torch.randint(0, 5, (4, 3))  # 4 edges, 3 features
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])  # Batch of 2
    
    model = EdgeAwareGIN()
    
    # Test graph-level output
    graph_out = model(batch)
    print(f"Graph output shape: {graph_out.shape}")  # Should be [2, 512]
    
    # Test node-level output
    node_out = model.get_node_embeddings(batch)
    print(f"Node output shape: {node_out.shape}")  # Should be [10, 512]
    
    print("EdgeAwareGIN test passed!")
