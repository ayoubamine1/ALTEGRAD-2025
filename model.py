"""
Complete Molecular Captioner model combining:
- EdgeAwareGIN graph encoder
- GraphToTextBridge projector  
- T5 decoder for text generation
"""
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

from graph_encoder import EdgeAwareGIN
from bridge import GraphToTextBridge, SimpleBridge


class MolecularCaptioner(nn.Module):
    """
    End-to-end Graph-to-Text model for molecular captioning.
    
    Architecture:
        Graph -> GIN Encoder -> Bridge Projector -> T5 Decoder -> Text
    """
    
    def __init__(
        self,
        lm_name: str = "laituan245/molt5-small",  # Chemistry-focused T5
        graph_hidden_dim: int = 256,
        graph_out_dim: int = 512,
        num_gnn_layers: int = 4,
        num_query_tokens: int = 32,
        use_simple_bridge: bool = False,
        freeze_lm: bool = True,
        gradient_checkpointing: bool = True
    ):
        """
        Args:
            lm_name: HuggingFace model name (t5-small, t5-base)
            graph_hidden_dim: Hidden dimension for GNN layers
            graph_out_dim: Output dimension of graph encoder
            num_gnn_layers: Number of GIN layers
            num_query_tokens: Number of tokens in bridge output
            use_simple_bridge: Use SimpleBridge instead of cross-attention
            freeze_lm: Freeze LM parameters initially
            gradient_checkpointing: Enable for memory efficiency
        """
        super().__init__()
        
        # Load T5 configuration to get dimensions
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        lm_dim = self.lm.config.d_model  # 512 for t5-small
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable()
        
        # Freeze LM if specified (unfreeze later for fine-tuning)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False
        
        # Graph encoder
        self.graph_encoder = EdgeAwareGIN(
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            num_layers=num_gnn_layers
        )
        
        # Bridge to project graph to LM space
        if use_simple_bridge:
            self.bridge = SimpleBridge(
                graph_dim=graph_out_dim,
                lm_dim=lm_dim,
                num_tokens=num_query_tokens
            )
            self.use_node_embeddings = False
        else:
            self.bridge = GraphToTextBridge(
                graph_dim=graph_out_dim,
                lm_dim=lm_dim,
                num_query_tokens=num_query_tokens
            )
            self.use_node_embeddings = True
        
        self.num_query_tokens = num_query_tokens
        self.lm_dim = lm_dim
        self.freeze_lm = freeze_lm
        
    def unfreeze_lm(self, unfreeze_layers: int = -1):
        """
        Unfreeze LM parameters for fine-tuning.
        
        Args:
            unfreeze_layers: Number of decoder layers to unfreeze.
                            -1 means unfreeze all.
        """
        if unfreeze_layers == -1:
            for param in self.lm.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only last N decoder layers
            decoder_layers = self.lm.decoder.block
            for layer in decoder_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Also unfreeze LM head
            for param in self.lm.lm_head.parameters():
                param.requires_grad = True
                
    def forward(
        self,
        graph_batch: Batch,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ):
        """
        Forward pass for training.
        
        Args:
            graph_batch: PyG Batch with molecular graphs
            labels: Target token IDs [B, seq_len]
            attention_mask: Not used (kept for API compatibility)
            
        Returns:
            loss: Cross-entropy loss if labels provided
            logits: Output logits [B, seq_len, vocab_size]
        """
        # Encode graph
        if self.use_node_embeddings:
            node_embs = self.graph_encoder.get_node_embeddings(graph_batch)
            graph_tokens = self.bridge(node_embs, graph_batch.batch)
        else:
            graph_emb = self.graph_encoder(graph_batch)
            graph_tokens = self.bridge(graph_emb)
        
        # graph_tokens: [B, num_query_tokens, lm_dim]
        batch_size = graph_tokens.size(0)
        
        # Create encoder outputs wrapper for T5
        # T5 expects BaseModelOutput with last_hidden_state
        encoder_outputs = BaseModelOutput(last_hidden_state=graph_tokens)
        
        # Create attention mask for encoder outputs (all valid)
        encoder_attention_mask = torch.ones(
            batch_size, self.num_query_tokens,
            device=graph_tokens.device,
            dtype=torch.long
        )
        
        # Forward through T5 decoder
        outputs = self.lm(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        graph_batch: Batch,
        max_length: int = 256,
        num_beams: int = 5,
        repetition_penalty: float = 1.5,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Generate captions for molecular graphs.
        
        Args:
            graph_batch: PyG Batch with molecular graphs
            max_length: Maximum generation length
            num_beams: Beam search width
            do_sample: Use sampling instead of beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            generated_ids: [B, seq_len] token IDs
        """
        self.eval()
        
        # Encode graph
        if self.use_node_embeddings:
            node_embs = self.graph_encoder.get_node_embeddings(graph_batch)
            graph_tokens = self.bridge(node_embs, graph_batch.batch)
        else:
            graph_emb = self.graph_encoder(graph_batch)
            graph_tokens = self.bridge(graph_emb)
        
        batch_size = graph_tokens.size(0)
        
        # Create encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=graph_tokens)
        encoder_attention_mask = torch.ones(
            batch_size, self.num_query_tokens,
            device=graph_tokens.device,
            dtype=torch.long
        )
        
        # Generate
        generated_ids = self.lm.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            early_stopping=True,
            **kwargs
        )
        
        return generated_ids
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(
    lm_name: str = "t5-small",
    freeze_lm: bool = True,
    use_simple_bridge: bool = False,
    gradient_checkpointing: bool = True
) -> MolecularCaptioner:
    """
    Factory function to create model with good defaults.
    """
    model = MolecularCaptioner(
        lm_name=lm_name,
        graph_hidden_dim=256,
        graph_out_dim=512,
        num_gnn_layers=4,
        num_query_tokens=8,
        use_simple_bridge=use_simple_bridge,
        freeze_lm=freeze_lm,
        gradient_checkpointing=gradient_checkpointing
    )
    
    trainable, total = model.count_parameters()
    print(f"Model created: {lm_name}")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  LM frozen: {freeze_lm}")
    
    return model


if __name__ == "__main__":
    print("Testing MolecularCaptioner...")
    
    from torch_geometric.data import Data
    
    # Create dummy graph data
    x = torch.randint(0, 10, (5, 9))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.randint(0, 5, (4, 3))
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])
    
    # Create model
    model = create_model(freeze_lm=True)
    
    # Test forward pass
    labels = torch.randint(0, 100, (2, 32))  # Dummy labels
    outputs = model(batch, labels=labels)
    print(f"Loss: {outputs.loss.item():.4f}")
    
    # Test generation
    generated = model.generate(batch, max_length=50, num_beams=2)
    print(f"Generated shape: {generated.shape}")
    
    print("\nMolecularCaptioner test passed!")
