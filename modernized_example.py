"""Modernized example script demonstrating GNN TSP optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.core import set_seed, get_device, generate_tsp_instance, create_graph_data, compute_tour_length, visualize_tour
from models.gnn_models import GraphPointerNetwork
from train.trainer import TSPTrainer, TSPDataset
from torch.utils.data import DataLoader


def main():
    """Main demonstration function."""
    print("GNN TSP Optimization Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate TSP instance
    print("\nGenerating TSP instance...")
    coords, edge_index = generate_tsp_instance(n_cities=15, seed=42)
    data = create_graph_data(coords, edge_index)
    print(f"Generated TSP with {coords.size(0)} cities")
    
    # Create model
    print("\nCreating Graph Pointer Network...")
    model = GraphPointerNetwork(
        input_dim=2,
        hidden_dim=128,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        use_gat=True
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    print("\nCreating training dataset...")
    dataset = TSPDataset(n_instances=100, n_cities=15, seed=42)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset size: {len(dataset)} instances")
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer = TSPTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
        use_wandb=False
    )
    
    # Train model
    print("\nTraining model...")
    print("This is a quick demo with limited training...")
    
    # Quick training loop (simplified)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):  # Just 5 epochs for demo
        total_loss = 0
        num_batches = 0
        
        for batch_data, batch_coords in dataloader:
            batch_data = batch_data.to(device)
            batch_coords = batch_coords.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            node_embeddings, attn_weights = model(batch_data)
            
            # Simple loss: minimize embedding distances
            loss = torch.mean(torch.norm(node_embeddings, dim=1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/5, Loss: {avg_loss:.4f}")
    
    # Test on the original instance
    print("\nTesting on original TSP instance...")
    model.eval()
    with torch.no_grad():
        node_embeddings, attn_weights = model(data.to(device))
        
        # Simple tour construction from embeddings
        def construct_tour_from_embeddings(embeddings):
            n = embeddings.size(0)
            visited = set()
            tour = []
            
            # Start from node with highest norm
            start = torch.norm(embeddings, dim=1).argmax().item()
            tour.append(start)
            visited.add(start)
            
            current = start
            for _ in range(n - 1):
                best_node = None
                best_score = float('-inf')
                
                for node in range(n):
                    if node not in visited:
                        # Combine embedding similarity with distance
                        embedding_sim = torch.cosine_similarity(
                            embeddings[current].unsqueeze(0), 
                            embeddings[node].unsqueeze(0)
                        ).item()
                        
                        if embedding_sim > best_score:
                            best_score = embedding_sim
                            best_node = node
                
                if best_node is not None:
                    tour.append(best_node)
                    visited.add(best_node)
                    current = best_node
            
            return tour
        
        gnn_tour = construct_tour_from_embeddings(node_embeddings)
        gnn_length = compute_tour_length(coords, gnn_tour)
        
        # Compare with greedy baseline
        def greedy_tour(coords):
            n = coords.size(0)
            visited = set()
            tour = []
            
            current = 0
            tour.append(current)
            visited.add(current)
            
            for _ in range(n - 1):
                best_city = None
                best_distance = float('inf')
                
                for city in range(n):
                    if city not in visited:
                        distance = torch.norm(coords[current] - coords[city]).item()
                        if distance < best_distance:
                            best_distance = distance
                            best_city = city
                
                if best_city is not None:
                    tour.append(best_city)
                    visited.add(best_city)
                    current = best_city
            
            return tour
        
        greedy_tour_result = greedy_tour(coords)
        greedy_length = compute_tour_length(coords, greedy_tour_result)
        
        print(f"GNN Tour: {gnn_tour}")
        print(f"GNN Tour Length: {gnn_length:.3f}")
        print(f"Greedy Tour: {greedy_tour_result}")
        print(f"Greedy Tour Length: {greedy_length:.3f}")
        print(f"Improvement: {((greedy_length - gnn_length) / greedy_length * 100):.1f}%")
    
    # Visualize results
    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GNN solution
    ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    for i in range(len(gnn_tour)):
        current = gnn_tour[i]
        next_node = gnn_tour[(i + 1) % len(gnn_tour)]
        ax1.plot([coords[current, 0], coords[next_node, 0]], 
                [coords[current, 1], coords[next_node, 1]], 'b-', alpha=0.7)
    ax1.set_title(f'GNN Solution (Length: {gnn_length:.3f})')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Greedy solution
    ax2.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    for i in range(len(greedy_tour_result)):
        current = greedy_tour_result[i]
        next_node = greedy_tour_result[(i + 1) % len(greedy_tour_result)]
        ax2.plot([coords[current, 0], coords[next_node, 0]], 
                [coords[current, 1], coords[next_node, 1]], 'b-', alpha=0.7)
    ax2.set_title(f'Greedy Solution (Length: {greedy_length:.3f})')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('gnn_tsp_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemo completed successfully!")
    print("Check 'gnn_tsp_demo.png' for the visualization.")


if __name__ == "__main__":
    main()
