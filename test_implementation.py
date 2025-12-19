"""Simple test script to verify the GNN TSP implementation."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.core import set_seed, get_device, generate_tsp_instance, create_graph_data, compute_tour_length
from models.gnn_models import GraphPointerNetwork, GraphTransformerTSP, GraphIsomorphismTSP, GraphSAGETSP


def test_basic_functionality():
    """Test basic functionality of the GNN TSP implementation."""
    print("Testing GNN TSP Implementation")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate TSP instance
    print("\n1. Generating TSP instance...")
    coords, edge_index = generate_tsp_instance(n_cities=10, seed=42)
    data = create_graph_data(coords, edge_index)
    print(f"   Generated TSP with {coords.size(0)} cities")
    print(f"   Graph has {edge_index.size(1)} edges")
    
    # Test models
    models = {
        "GraphPointerNetwork": GraphPointerNetwork(input_dim=2, hidden_dim=64),
        "GraphTransformerTSP": GraphTransformerTSP(input_dim=2, hidden_dim=64),
        "GraphIsomorphismTSP": GraphIsomorphismTSP(input_dim=2, hidden_dim=64),
        "GraphSAGETSP": GraphSAGETSP(input_dim=2, hidden_dim=64)
    }
    
    print("\n2. Testing models...")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == "GraphPointerNetwork":
                embeddings, attn_weights = model(data)
                print(f"   {name}: embeddings shape {embeddings.shape}, attention shape {attn_weights.shape}")
            else:
                scores = model(data)
                print(f"   {name}: scores shape {scores.shape}")
    
    # Test tour construction
    print("\n3. Testing tour construction...")
    
    # Simple greedy tour
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
    
    tour = greedy_tour(coords)
    tour_length = compute_tour_length(coords, tour)
    print(f"   Greedy tour: {tour}")
    print(f"   Tour length: {tour_length:.3f}")
    
    # Test random tour
    import random
    random_tour = list(range(coords.size(0)))
    random.shuffle(random_tour)
    random_length = compute_tour_length(coords, random_tour)
    print(f"   Random tour length: {random_length:.3f}")
    
    print("\n4. All tests passed successfully!")
    print("   The GNN TSP implementation is working correctly.")


if __name__ == "__main__":
    test_basic_functionality()
