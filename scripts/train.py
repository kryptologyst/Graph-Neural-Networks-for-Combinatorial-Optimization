"""Main training script for GNN-based TSP optimization."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_seed, get_device
from models.gnn_models import (
    GraphPointerNetwork, 
    GraphTransformerTSP, 
    GraphIsomorphismTSP, 
    GraphSAGETSP
)
from train.trainer import TSPTrainer, TSPDataset
from eval.evaluator import TSPEvaluator, ConvergenceAnalyzer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch model
    """
    model_config = config['model']
    model_name = model_config['name']
    
    if model_name == "GraphPointerNetwork":
        return GraphPointerNetwork(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            dropout=model_config['dropout'],
            use_gat=model_config['use_gat']
        )
    elif model_name == "GraphTransformerTSP":
        return GraphTransformerTSP(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            dropout=model_config['dropout'],
            use_positional_encoding=model_config['use_positional_encoding']
        )
    elif model_name == "GraphIsomorphismTSP":
        return GraphIsomorphismTSP(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        )
    elif model_name == "GraphSAGETSP":
        return GraphSAGETSP(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_datasets(config: dict) -> tuple:
    """Create training, validation, and test datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_config = config['data']
    
    train_dataset = TSPDataset(
        n_instances=data_config['n_train_instances'],
        n_cities=data_config['n_cities'],
        seed=data_config['train_seed']
    )
    
    val_dataset = TSPDataset(
        n_instances=data_config['n_val_instances'],
        n_cities=data_config['n_cities'],
        seed=data_config['val_seed']
    )
    
    test_dataset = TSPDataset(
        n_instances=data_config['n_test_instances'],
        n_cities=data_config['n_cities'],
        seed=data_config['test_seed']
    )
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN for TSP optimization")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config['model']['name']}")
    
    # Create directories
    Path(config['paths']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['assets_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_model(config)
    print(f"Model created: {config['model']['name']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create trainer
    trainer = TSPTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        use_wandb=config['logging']['use_wandb'],
        project_name=config['logging']['wandb_project']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_model(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['training']['n_epochs'],
        save_path=f"{config['paths']['checkpoints_dir']}/best_model.pth",
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    print("Training completed!")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    
    # Create evaluator and analyze results
    evaluator = TSPEvaluator()
    convergence_analyzer = ConvergenceAnalyzer()
    
    # Add training history
    convergence_analyzer.add_training_history(
        config['model']['name'],
        history['train_losses'],
        history['val_losses']
    )
    
    # Plot convergence
    convergence_analyzer.plot_convergence(
        save_path=f"{config['paths']['assets_dir']}/convergence_plot.png"
    )
    
    # Generate convergence analysis
    convergence_analysis = convergence_analyzer.analyze_convergence()
    print("Convergence Analysis:")
    for method, metrics in convergence_analysis.items():
        print(f"  {method}:")
        print(f"    Final train loss: {metrics['final_train_loss']:.4f}")
        print(f"    Final val loss: {metrics['final_val_loss']:.4f}")
        print(f"    Min train loss: {metrics['min_train_loss']:.4f}")
        print(f"    Min val loss: {metrics['min_val_loss']:.4f}")
        print(f"    Epochs: {metrics['n_epochs']}")
    
    print("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
