"""
Bridge module to project graph representations into LLM embedding space.
Uses learnable query tokens with cross-attention (Q-Former style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphToTextBridge(nn.Module):
    """
    Projects graph node embeddings to a fixed-length sequence of
    "graph tokens" that can be consumed by a language model.
    
    Uses learnable query tokens that attend to graph node embeddings
    via cross-attention, similar to Q-Former in BLIP-2.
    """
    
    def __init__(
        self,
        graph_dim: int = 512,
        lm_dim: int = 512,  # T5-small hidden size
        num_query_tokens: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            graph_dim: Dimension of graph node embeddings
            lm_dim: Target dimension for LM (T5-small=512, T5-base=768)
            num_query_tokens: Number of output tokens to produce
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.graph_dim = graph_dim
        self.lm_dim = lm_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, graph_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Cross-attention layers (queries attend to graph nodes)
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=graph_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(graph_dim, graph_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(graph_dim * 4, graph_dim),
                    nn.Dropout(dropout)
                )
            )
            self.norm1_layers.append(nn.LayerNorm(graph_dim))
            self.norm2_layers.append(nn.LayerNorm(graph_dim))
        
        # Final projection to LM dimension
        self.output_proj = nn.Sequential(
            nn.Linear(graph_dim, lm_dim),
            nn.LayerNorm(lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, graph_dim] embeddings for all nodes in batch
            batch_indices: [N] tensor indicating which graph each node belongs to
            
        Returns:
            graph_tokens: [B, num_query_tokens, lm_dim] output for LM
        """
        # Get batch size
        batch_size = batch_indices.max().item() + 1
        device = node_embeddings.device
        
        # Expand query tokens for batch: [B, num_query_tokens, graph_dim]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create padded node embeddings per graph
        # Find max nodes in this batch
        nodes_per_graph = torch.bincount(batch_indices, minlength=batch_size)
        max_nodes = nodes_per_graph.max().item()
        
        # Create padded tensor [B, max_nodes, graph_dim]
        padded_nodes = torch.zeros(
            batch_size, max_nodes, self.graph_dim,
            device=device, dtype=node_embeddings.dtype
        )
        
        # Create attention mask [B, max_nodes] (True = masked/invalid)
        key_padding_mask = torch.ones(
            batch_size, max_nodes,
            device=device, dtype=torch.bool
        )
        
        # Fill in the actual node embeddings
        for b in range(batch_size):
            mask = (batch_indices == b)
            n_nodes = mask.sum().item()
            padded_nodes[b, :n_nodes] = node_embeddings[mask]
            key_padding_mask[b, :n_nodes] = False
        
        # Cross-attention layers
        h = queries
        for cross_attn, ffn, norm1, norm2 in zip(
            self.cross_attn_layers,
            self.ffn_layers,
            self.norm1_layers,
            self.norm2_layers
        ):
            # Cross-attention (queries attend to nodes)
            h_attn, _ = cross_attn(
                query=h,
                key=padded_nodes,
                value=padded_nodes,
                key_padding_mask=key_padding_mask
            )
            h = norm1(h + self.dropout(h_attn))
            
            # Feed-forward
            h = norm2(h + ffn(h))
        
        # Project to LM dimension
        output = self.output_proj(h)  # [B, num_query_tokens, lm_dim]
        
        return output


class SimpleBridge(nn.Module):
    """
    Simpler bridge that just uses global pooled graph embedding.
    Projects to a single "prefix token" for the LM.
    
    Use this if GraphToTextBridge is too slow or memory-intensive.
    """
    
    def __init__(
        self,
        graph_dim: int = 512,
        lm_dim: int = 512,
        num_tokens: int = 8
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        
        # Project graph embedding to multiple tokens
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, lm_dim * num_tokens),
            nn.GELU(),
            nn.Linear(lm_dim * num_tokens, lm_dim * num_tokens)
        )
        self.norm = nn.LayerNorm(lm_dim)
        self.lm_dim = lm_dim
        
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embedding: [B, graph_dim] global graph embeddings
            
        Returns:
            tokens: [B, num_tokens, lm_dim]
        """
        batch_size = graph_embedding.size(0)
        h = self.proj(graph_embedding)  # [B, lm_dim * num_tokens]
        h = h.view(batch_size, self.num_tokens, self.lm_dim)
        h = self.norm(h)
        return h


if __name__ == "__main__":
    print("Testing GraphToTextBridge...")
    
    # Simulate node embeddings from GNN
    batch_size = 4
    nodes_per_graph = [5, 8, 3, 6]  # Variable nodes per graph
    total_nodes = sum(nodes_per_graph)
    graph_dim = 512
    
    node_embs = torch.randn(total_nodes, graph_dim)
    batch_indices = torch.cat([
        torch.full((n,), i) for i, n in enumerate(nodes_per_graph)
    ])
    
    bridge = GraphToTextBridge(graph_dim=512, lm_dim=512, num_query_tokens=8)
    output = bridge(node_embs, batch_indices)
    
    print(f"Input: {total_nodes} nodes across {batch_size} graphs")
    print(f"Output shape: {output.shape}")  # [4, 8, 512]
    assert output.shape == (batch_size, 8, 512)
    
    print("\nTesting SimpleBridge...")
    graph_emb = torch.randn(batch_size, graph_dim)
    simple_bridge = SimpleBridge(graph_dim=512, lm_dim=512, num_tokens=8)
    simple_out = simple_bridge(graph_emb)
    print(f"Output shape: {simple_out.shape}")  # [4, 8, 512]
    
    print("\nBridge tests passed!")
