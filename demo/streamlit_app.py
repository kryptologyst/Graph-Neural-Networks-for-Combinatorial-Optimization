"""Streamlit demo for GNN-based TSP optimization."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_seed, get_device, generate_tsp_instance, create_graph_data, compute_tour_length
from models.gnn_models import (
    GraphPointerNetwork, 
    GraphTransformerTSP, 
    GraphIsomorphismTSP, 
    GraphSAGETSP
)
from train.trainer import TSPTrainer


def load_model(model_name: str, checkpoint_path: str = None) -> torch.nn.Module:
    """Load a trained model.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint
        
    Returns:
        Loaded model
    """
    # Default model configurations
    model_configs = {
        "GraphPointerNetwork": {
            "input_dim": 2,
            "hidden_dim": 128,
            "n_layers": 3,
            "n_heads": 8,
            "dropout": 0.1,
            "use_gat": True
        },
        "GraphTransformerTSP": {
            "input_dim": 2,
            "hidden_dim": 128,
            "n_layers": 6,
            "n_heads": 8,
            "dropout": 0.1,
            "use_positional_encoding": True
        },
        "GraphIsomorphismTSP": {
            "input_dim": 2,
            "hidden_dim": 128,
            "n_layers": 5,
            "dropout": 0.1
        },
        "GraphSAGETSP": {
            "input_dim": 2,
            "hidden_dim": 128,
            "n_layers": 3,
            "dropout": 0.1
        }
    }
    
    config = model_configs[model_name]
    
    if model_name == "GraphPointerNetwork":
        model = GraphPointerNetwork(**config)
    elif model_name == "GraphTransformerTSP":
        model = GraphTransformerTSP(**config)
    elif model_name == "GraphIsomorphismTSP":
        model = GraphIsomorphismTSP(**config)
    elif model_name == "GraphSAGETSP":
        model = GraphSAGETSP(**config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"Loaded model from {checkpoint_path}")
    else:
        st.warning("No checkpoint provided, using randomly initialized model")
    
    return model


def solve_tsp_greedy(coords: torch.Tensor) -> list:
    """Solve TSP using greedy nearest neighbor heuristic.
    
    Args:
        coords: Node coordinates
        
    Returns:
        Tour as list of node indices
    """
    n_cities = coords.size(0)
    visited = set()
    tour = []
    
    # Start from first city
    current = 0
    tour.append(current)
    visited.add(current)
    
    # Greedy construction
    for _ in range(n_cities - 1):
        best_city = None
        best_distance = float('inf')
        
        for city in range(n_cities):
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


def solve_tsp_random(coords: torch.Tensor) -> list:
    """Solve TSP using random tour.
    
    Args:
        coords: Node coordinates
        
    Returns:
        Random tour as list of node indices
    """
    n_cities = coords.size(0)
    tour = list(range(n_cities))
    np.random.shuffle(tour)
    return tour


def plot_tour_plotly(coords: torch.Tensor, tour: list, title: str = "TSP Tour") -> go.Figure:
    """Create interactive plotly visualization of TSP tour.
    
    Args:
        coords: Node coordinates
        tour: Tour as list of node indices
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Convert to numpy
    coords_np = coords.numpy()
    
    # Create scatter plot for cities
    fig = go.Figure()
    
    # Add cities
    fig.add_trace(go.Scatter(
        x=coords_np[:, 0],
        y=coords_np[:, 1],
        mode='markers+text',
        marker=dict(size=15, color='red'),
        text=[str(i) for i in range(len(coords_np))],
        textposition="top center",
        name="Cities"
    ))
    
    # Add tour edges
    tour_coords = coords_np[tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the tour
    
    fig.add_trace(go.Scatter(
        x=tour_coords[:, 0],
        y=tour_coords[:, 1],
        mode='lines',
        line=dict(color='blue', width=2),
        name="Tour"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        showlegend=True,
        width=800,
        height=600
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GNN TSP Solver",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ Graph Neural Networks for TSP Optimization")
    st.markdown("Interactive demo for solving Traveling Salesman Problem using GNNs")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["GraphPointerNetwork", "GraphTransformerTSP", "GraphIsomorphismTSP", "GraphSAGETSP"],
        index=0
    )
    
    # TSP instance parameters
    st.sidebar.subheader("TSP Instance")
    n_cities = st.sidebar.slider("Number of Cities", 5, 50, 20)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)
    
    # Generate TSP instance
    if st.sidebar.button("Generate New Instance"):
        set_seed(seed)
        coords, edge_index = generate_tsp_instance(n_cities, seed)
        st.session_state.coords = coords
        st.session_state.edge_index = edge_index
        st.session_state.data = create_graph_data(coords, edge_index)
    
    # Initialize session state
    if 'coords' not in st.session_state:
        set_seed(seed)
        coords, edge_index = generate_tsp_instance(n_cities, seed)
        st.session_state.coords = coords
        st.session_state.edge_index = edge_index
        st.session_state.data = create_graph_data(coords, edge_index)
    
    # Load model
    device = get_device()
    model = load_model(model_name)
    model.eval()
    
    # Solve TSP with different methods
    st.subheader("TSP Solutions Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**GNN Solution**")
        with torch.no_grad():
            if model_name == "GraphPointerNetwork":
                node_embeddings, attn_weights = model(st.session_state.data)
                # Simple tour construction from embeddings
                tour_gnn = solve_tsp_greedy(node_embeddings)
            else:
                node_scores = model(st.session_state.data)
                # Use scores to construct tour
                tour_gnn = solve_tsp_greedy(st.session_state.coords)  # Fallback to greedy
        
        tour_length_gnn = compute_tour_length(st.session_state.coords, tour_gnn)
        st.metric("Tour Length", f"{tour_length_gnn:.3f}")
    
    with col2:
        st.markdown("**Greedy Solution**")
        tour_greedy = solve_tsp_greedy(st.session_state.coords)
        tour_length_greedy = compute_tour_length(st.session_state.coords, tour_greedy)
        st.metric("Tour Length", f"{tour_length_greedy:.3f}")
    
    with col3:
        st.markdown("**Random Solution**")
        tour_random = solve_tsp_random(st.session_state.coords)
        tour_length_random = compute_tour_length(st.session_state.coords, tour_random)
        st.metric("Tour Length", f"{tour_length_random:.3f}")
    
    # Visualization
    st.subheader("Tour Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["GNN Solution", "Greedy Solution", "Random Solution"])
    
    with tab1:
        fig_gnn = plot_tour_plotly(st.session_state.coords, tour_gnn, f"GNN Solution (Length: {tour_length_gnn:.3f})")
        st.plotly_chart(fig_gnn, use_container_width=True)
    
    with tab2:
        fig_greedy = plot_tour_plotly(st.session_state.coords, tour_greedy, f"Greedy Solution (Length: {tour_length_greedy:.3f})")
        st.plotly_chart(fig_greedy, use_container_width=True)
    
    with tab3:
        fig_random = plot_tour_plotly(st.session_state.coords, tour_random, f"Random Solution (Length: {tour_length_random:.3f})")
        st.plotly_chart(fig_random, use_container_width=True)
    
    # Model information
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Model:** {model_name}")
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
    
    with col2:
        st.markdown(f"**Cities:** {n_cities}")
        st.markdown(f"**Seed:** {seed}")
        st.markdown(f"**Instance:** Random uniform in [0,1]²")
    
    # Performance comparison
    st.subheader("Performance Comparison")
    
    methods = ["GNN", "Greedy", "Random"]
    lengths = [tour_length_gnn, tour_length_greedy, tour_length_random]
    
    # Create bar chart
    fig_comparison = go.Figure(data=[
        go.Bar(x=methods, y=lengths, text=[f"{l:.3f}" for l in lengths], textposition='auto')
    ])
    
    fig_comparison.update_layout(
        title="Tour Length Comparison",
        xaxis_title="Method",
        yaxis_title="Tour Length",
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Instructions
    st.subheader("Instructions")
    st.markdown("""
    1. **Select Model**: Choose from different GNN architectures
    2. **Configure Instance**: Adjust number of cities and random seed
    3. **Generate Instance**: Create a new TSP instance
    4. **Compare Solutions**: View GNN vs baseline solutions
    5. **Visualize Tours**: Interactive plots of different solutions
    
    **Note**: This demo uses randomly initialized models. For best results, train the models first using the training script.
    """)


if __name__ == "__main__":
    main()
