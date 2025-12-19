"""Training framework for GNN-based combinatorial optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from ..utils.core import set_seed, get_device, compute_tour_length, EarlyStopping
from ..models.gnn_models import GraphPointerNetwork, GraphTransformerTSP, GraphIsomorphismTSP


class TSPDataset(Dataset):
    """Dataset for TSP instances."""
    
    def __init__(self, n_instances: int = 1000, n_cities: int = 20, seed: Optional[int] = None):
        """Initialize TSP dataset.
        
        Args:
            n_instances: Number of TSP instances to generate
            n_cities: Number of cities per instance
            seed: Random seed
        """
        self.n_instances = n_instances
        self.n_cities = n_cities
        self.seed = seed
        
        # Generate instances
        self.instances = []
        for i in range(n_instances):
            if seed is not None:
                set_seed(seed + i)
            
            from ..utils.core import generate_tsp_instance, create_graph_data
            coords, edge_index = generate_tsp_instance(n_cities)
            data = create_graph_data(coords, edge_index)
            self.instances.append((data, coords))
    
    def __len__(self) -> int:
        return self.n_instances
    
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        return self.instances[idx]


class TSPTrainer:
    """Trainer for TSP GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_wandb: bool = False,
        project_name: str = "gnn-tsp"
    ):
        """Initialize trainer.
        
        Args:
            model: GNN model
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            use_wandb: Whether to use Weights & Biases
            project_name: W&B project name
        """
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Initialize W&B
        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(model)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_data, batch_coords in pbar:
            batch_data = batch_data.to(self.device)
            batch_coords = batch_coords.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, GraphPointerNetwork):
                node_embeddings, attn_weights = self.model(batch_data)
                # Use attention weights for tour construction
                tour = self._construct_tour_from_attention(attn_weights, batch_coords)
            else:
                node_scores = self.model(batch_data)
                tour = self._construct_tour_from_scores(node_scores, batch_coords)
            
            # Compute loss
            loss = self._compute_tsp_loss(batch_coords, tour)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        if self.use_wandb:
            wandb.log({"train_loss": avg_loss})
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch_data, batch_coords in pbar:
                batch_data = batch_data.to(self.device)
                batch_coords = batch_coords.to(self.device)
                
                # Forward pass
                if isinstance(self.model, GraphPointerNetwork):
                    node_embeddings, attn_weights = self.model(batch_data)
                    tour = self._construct_tour_from_attention(attn_weights, batch_coords)
                else:
                    node_scores = self.model(batch_data)
                    tour = self._construct_tour_from_scores(node_scores, batch_coords)
                
                # Compute loss
                loss = self._compute_tsp_loss(batch_coords, tour)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        if self.use_wandb:
            wandb.log({"val_loss": avg_loss})
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            save_path: Path to save best model
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path:
                    self.save_model(save_path)
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
    
    def _construct_tour_from_scores(self, scores: torch.Tensor, coords: torch.Tensor) -> List[int]:
        """Construct tour from node scores using greedy selection.
        
        Args:
            scores: Node scores [n_nodes]
            coords: Node coordinates [n_nodes, 2]
            
        Returns:
            Tour as list of node indices
        """
        n_nodes = scores.size(0)
        visited = set()
        tour = []
        
        # Start from highest scoring node
        start_node = scores.argmax().item()
        tour.append(start_node)
        visited.add(start_node)
        
        # Greedy construction
        current = start_node
        for _ in range(n_nodes - 1):
            best_node = None
            best_score = float('-inf')
            
            for node in range(n_nodes):
                if node not in visited:
                    # Combine score with distance heuristic
                    distance = torch.norm(coords[current] - coords[node]).item()
                    combined_score = scores[node].item() - 0.1 * distance  # Penalize distance
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_node = node
            
            if best_node is not None:
                tour.append(best_node)
                visited.add(best_node)
                current = best_node
        
        return tour
    
    def _construct_tour_from_attention(self, attn_weights: torch.Tensor, coords: torch.Tensor) -> List[int]:
        """Construct tour from attention weights.
        
        Args:
            attn_weights: Attention weights [n_nodes, n_nodes]
            coords: Node coordinates [n_nodes, 2]
            
        Returns:
            Tour as list of node indices
        """
        n_nodes = attn_weights.size(0)
        visited = set()
        tour = []
        
        # Start from node with highest self-attention
        start_node = torch.diag(attn_weights).argmax().item()
        tour.append(start_node)
        visited.add(start_node)
        
        # Greedy construction based on attention
        current = start_node
        for _ in range(n_nodes - 1):
            best_node = None
            best_attention = float('-inf')
            
            for node in range(n_nodes):
                if node not in visited:
                    attention_score = attn_weights[current, node].item()
                    if attention_score > best_attention:
                        best_attention = attention_score
                        best_node = node
            
            if best_node is not None:
                tour.append(best_node)
                visited.add(best_node)
                current = best_node
        
        return tour
    
    def _compute_tsp_loss(self, coords: torch.Tensor, tour: List[int]) -> torch.Tensor:
        """Compute TSP tour length loss.
        
        Args:
            coords: Node coordinates [n_nodes, 2]
            tour: Tour as list of node indices
            
        Returns:
            Tour length loss
        """
        if len(tour) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        total_length = 0.0
        for i in range(len(tour)):
            current = tour[i]
            next_node = tour[(i + 1) % len(tour)]
            total_length += torch.norm(coords[current] - coords[next_node]).item()
        
        return torch.tensor(total_length, device=self.device, requires_grad=True)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Model loaded from {path}")
