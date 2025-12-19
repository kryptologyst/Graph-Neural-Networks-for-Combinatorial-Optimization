"""Advanced GNN models for combinatorial optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax
import math


class GraphPointerNetwork(nn.Module):
    """Graph Pointer Network for TSP solving with attention-based decoding."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gat: bool = True
    ):
        """Initialize Graph Pointer Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of GNN layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_gat: Whether to use GAT or GCN layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_gat = use_gat
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_gat:
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim // n_heads, 
                        heads=n_heads,
                        dropout=dropout,
                        concat=True if i < n_layers - 1 else False
                    )
                )
            else:
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention mechanism for pointer network
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Output projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tuple of (node_embeddings, attention_weights)
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        h = self.input_proj(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            if self.use_gat:
                h = layer(h, edge_index)
            else:
                h = F.relu(layer(h, edge_index))
            h = self.dropout(h)
        
        # Compute attention weights for pointer network
        query = self.query_proj(h)  # [n_nodes, hidden_dim]
        key = self.key_proj(h)      # [n_nodes, hidden_dim]
        value = self.value_proj(h)  # [n_nodes, hidden_dim]
        
        # Reshape for multi-head attention
        query = query.unsqueeze(0)  # [1, n_nodes, hidden_dim]
        key = key.unsqueeze(0)      # [1, n_nodes, hidden_dim]
        value = value.unsqueeze(0)  # [1, n_nodes, hidden_dim]
        
        attn_output, attn_weights = self.attention(query, key, value)
        attn_output = attn_output.squeeze(0)  # [n_nodes, hidden_dim]
        
        return h, attn_weights.squeeze(0)  # [n_nodes, n_nodes]


class GraphAttentionDecoder(nn.Module):
    """Attention-based decoder for constructing TSP tours."""
    
    def __init__(self, hidden_dim: int = 128, n_heads: int = 8):
        """Initialize decoder.
        
        Args:
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Attention layers
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            node_embeddings: Node embeddings [n_nodes, hidden_dim]
            mask: Attention mask [n_nodes, n_nodes]
            
        Returns:
            Attention scores [n_nodes, 1]
        """
        # Add batch dimension
        embeddings = node_embeddings.unsqueeze(0)  # [1, n_nodes, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.attention(embeddings, embeddings, embeddings, attn_mask=mask)
        
        # Remove batch dimension
        attn_output = attn_output.squeeze(0)  # [n_nodes, hidden_dim]
        
        # Output projection
        scores = self.output_proj(attn_output)  # [n_nodes, 1]
        
        return scores.squeeze(-1)  # [n_nodes]


class GraphTransformerTSP(nn.Module):
    """Graph Transformer for TSP solving with positional encoding."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """Initialize Graph Transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))  # Max 1000 nodes
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node scores [n_nodes]
        """
        x = data.x
        
        # Input projection
        h = self.input_proj(x)  # [n_nodes, hidden_dim]
        
        # Add positional encoding
        if self.use_positional_encoding and h.size(0) <= self.pos_encoding.size(0):
            h = h + self.pos_encoding[:h.size(0)]
        
        # Add batch dimension
        h = h.unsqueeze(0)  # [1, n_nodes, hidden_dim]
        
        # Transformer
        h = self.transformer(h)  # [1, n_nodes, hidden_dim]
        
        # Remove batch dimension
        h = h.squeeze(0)  # [n_nodes, hidden_dim]
        
        # Output projection
        scores = self.output_proj(h)  # [n_nodes, 1]
        
        return scores.squeeze(-1)  # [n_nodes]


class GraphIsomorphismTSP(nn.Module):
    """Graph Isomorphism Network for TSP solving."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 5,
        dropout: float = 0.1
    ):
        """Initialize GIN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of GIN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        
        # First layer
        self.gin_layers.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        )
        
        # Remaining layers
        for _ in range(n_layers - 1):
            self.gin_layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node scores [n_nodes]
        """
        x, edge_index = data.x, data.edge_index
        
        # GIN layers
        h = x
        for layer in self.gin_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Output projection
        scores = self.output_proj(h)  # [n_nodes, 1]
        
        return scores.squeeze(-1)  # [n_nodes]


class GraphSAGETSP(nn.Module):
    """GraphSAGE for TSP solving with neighbor sampling."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize GraphSAGE.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of GraphSAGE layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # GraphSAGE layers (simplified version)
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Remaining layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node scores [n_nodes]
        """
        x, edge_index = data.x, data.edge_index
        
        # Simple message passing (can be enhanced with proper GraphSAGE)
        h = x
        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Output projection
        scores = self.output_proj(h)  # [n_nodes, 1]
        
        return scores.squeeze(-1)  # [n_nodes]
