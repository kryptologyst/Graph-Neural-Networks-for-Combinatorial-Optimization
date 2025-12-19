#!/usr/bin/env python3
"""Setup script for GNN TSP optimization project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("GNN TSP Optimization Project Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    print("\nCreating project directories...")
    directories = [
        "data",
        "assets",
        "checkpoints", 
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    if not run_command("python test_implementation.py", "Running basic tests"):
        print("Basic tests failed")
        sys.exit(1)
    
    # Install pre-commit hooks (optional)
    print("\nSetting up pre-commit hooks...")
    if run_command("pip install pre-commit", "Installing pre-commit"):
        if run_command("pre-commit install", "Installing pre-commit hooks"):
            print("✓ Pre-commit hooks installed")
        else:
            print("! Pre-commit hooks installation failed (optional)")
    else:
        print("! Pre-commit installation failed (optional)")
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/streamlit_app.py")
    print("2. Train a model: python scripts/train.py --config configs/config.yaml")
    print("3. Try the modernized example: python modernized_example.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
