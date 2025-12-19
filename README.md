# Graph Neural Networks for Combinatorial Optimization

A production-ready implementation of Graph Neural Networks (GNNs) for solving the Traveling Salesman Problem (TSP) and other combinatorial optimization tasks.

## Overview

This project implements state-of-the-art GNN architectures for combinatorial optimization, focusing on the Traveling Salesman Problem as a case study. The framework includes multiple GNN models, comprehensive evaluation metrics, and interactive visualization tools.

## Features

- **Multiple GNN Architectures**: Graph Pointer Networks, Graph Transformers, Graph Isomorphism Networks (GIN), and GraphSAGE
- **Modern ML Stack**: PyTorch 2.x, PyTorch Geometric, Weights & Biases integration
- **Comprehensive Evaluation**: Tour length metrics, optimality gap analysis, convergence tracking
- **Interactive Demo**: Streamlit application for real-time TSP solving
- **Production Ready**: Type hints, configuration management, checkpointing, logging

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Graph-Neural-Networks-for-Combinatorial-Optimization.git
cd Graph-Neural-Networks-for-Combinatorial-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

### Training

Train a GNN model on TSP instances:

```bash
python scripts/train.py --config configs/config.yaml --device auto
```

### Configuration

Modify `configs/config.yaml` to customize:
- Model architecture and hyperparameters
- Training settings
- Data generation parameters
- Evaluation metrics

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # GNN model implementations
│   ├── train/             # Training framework
│   ├── eval/              # Evaluation metrics and analysis
│   └── utils/             # Core utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Interactive Streamlit demo
├── data/                  # Data storage
├── assets/                # Generated plots and artifacts
├── checkpoints/           # Model checkpoints
└── tests/                 # Unit tests
```

## Models

### Graph Pointer Network
- Attention-based architecture for sequential decision making
- Multi-head attention mechanism for tour construction
- Suitable for TSP and similar routing problems

### Graph Transformer
- Global attention mechanism with positional encoding
- State-of-the-art performance on graph tasks
- Configurable number of layers and attention heads

### Graph Isomorphism Network (GIN)
- Powerful graph representation learning
- Excellent for molecular and structural graphs
- Sum aggregation with MLP updates

### GraphSAGE
- Inductive learning on large graphs
- Neighbor sampling for scalability
- Mean/pool/LSTM aggregation options

## Usage Examples

### Basic Training

```python
from src.models.gnn_models import GraphPointerNetwork
from src.train.trainer import TSPTrainer
from src.utils.core import get_device

# Create model
model = GraphPointerNetwork(input_dim=2, hidden_dim=128)

# Create trainer
device = get_device()
trainer = TSPTrainer(model, device)

# Train model
history = trainer.train(train_loader, val_loader, n_epochs=100)
```

### Evaluation

```python
from src.eval.evaluator import TSPEvaluator

# Create evaluator
evaluator = TSPEvaluator()

# Evaluate tour
metrics = evaluator.evaluate_tour(coords, predicted_tour, optimal_tour)
print(f"Tour length: {metrics['tour_length']:.3f}")
print(f"Optimality gap: {metrics['optimality_gap']:.2f}%")
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/streamlit_app.py
```

Features:
- Real-time TSP instance generation
- Multiple GNN model comparison
- Interactive tour visualization
- Performance metrics dashboard

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
model:
  name: "GraphPointerNetwork"
  hidden_dim: 128
  n_layers: 3
  n_heads: 8

training:
  batch_size: 32
  learning_rate: 0.001
  n_epochs: 100

data:
  n_cities: 20
  n_train_instances: 1000
```

## Evaluation Metrics

- **Tour Length**: Total distance of the constructed tour
- **Optimality Gap**: Percentage deviation from optimal solution
- **Convergence Analysis**: Training and validation loss tracking
- **Method Comparison**: Side-by-side performance comparison

## Advanced Features

### Device Support
- Automatic device detection (CUDA → MPS → CPU)
- Mixed precision training support
- Multi-GPU training ready

### Logging and Monitoring
- Weights & Biases integration
- TensorBoard support
- Comprehensive checkpointing

### Reproducibility
- Deterministic seeding
- Configuration versioning
- Experiment tracking

## Performance

Typical performance on TSP instances:
- **20 cities**: GNN solutions within 5-15% of optimal
- **50 cities**: Scalable with neighbor sampling
- **Training time**: ~10 minutes on modern GPU

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_tsp_optimization,
  title={Graph Neural Networks for Combinatorial Optimization},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Graph-Neural-Networks-for-Combinatorial-Optimization}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- Original TSP and GNN research papers
- Open source community for tools and libraries

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Import errors**: Ensure all dependencies are installed correctly
3. **Model not loading**: Check checkpoint path and model architecture match

### Getting Help

- Check the issues section for common problems
- Review the configuration examples
- Run tests to verify installation

## Future Work

- [ ] Support for other combinatorial optimization problems (VRP, CVRP)
- [ ] Reinforcement learning integration
- [ ] Multi-objective optimization
- [ ] Real-world dataset integration
- [ ] Model compression and quantization
# Graph-Neural-Networks-for-Combinatorial-Optimization
