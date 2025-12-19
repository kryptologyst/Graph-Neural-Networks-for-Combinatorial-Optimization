"""Core utilities for GNN-based combinatorial optimization."""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_tour_length(coords: torch.Tensor, tour: List[int]) -> float:
    """Compute the total length of a TSP tour.
    
    Args:
        coords: Node coordinates [n_nodes, 2]
        tour: List of node indices forming the tour
        
    Returns:
        Total tour length
    """
    if len(tour) <= 1:
        return 0.0
    
    total_length = 0.0
    for i in range(len(tour)):
        current = tour[i]
        next_node = tour[(i + 1) % len(tour)]
        total_length += torch.norm(coords[current] - coords[next_node]).item()
    
    return total_length


def compute_optimality_gap(heuristic_length: float, optimal_length: float) -> float:
    """Compute optimality gap percentage.
    
    Args:
        heuristic_length: Length found by heuristic
        optimal_length: Optimal tour length
        
    Returns:
        Optimality gap as percentage
    """
    if optimal_length == 0:
        return float('inf')
    return ((heuristic_length - optimal_length) / optimal_length) * 100


def generate_tsp_instance(n_cities: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random TSP instance.
    
    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (coordinates, edge_index) for the graph
    """
    if seed is not None:
        set_seed(seed)
    
    # Generate random coordinates in [0, 1] x [0, 1]
    coords = torch.rand(n_cities, 2)
    
    # Create fully connected undirected graph
    edge_index = torch.combinations(torch.arange(n_cities), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return coords, edge_index


def create_graph_data(coords: torch.Tensor, edge_index: torch.Tensor) -> Data:
    """Create PyTorch Geometric Data object from coordinates and edges.
    
    Args:
        coords: Node coordinates [n_nodes, 2]
        edge_index: Edge connectivity [2, n_edges]
        
    Returns:
        PyTorch Geometric Data object
    """
    return Data(x=coords, edge_index=edge_index)


def visualize_tour(coords: torch.Tensor, tour: List[int], title: str = "TSP Tour") -> None:
    """Visualize a TSP tour.
    
    Args:
        coords: Node coordinates [n_nodes, 2]
        tour: List of node indices forming the tour
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    
    # Plot tour edges
    for i in range(len(tour)):
        current = tour[i]
        next_node = tour[(i + 1) % len(tour)]
        plt.plot([coords[current, 0], coords[next_node, 0]], 
                [coords[current, 1], coords[next_node, 1]], 'b-', alpha=0.7)
    
    # Add city labels
    for i, coord in enumerate(coords):
        plt.annotate(str(i), (coord[0], coord[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
