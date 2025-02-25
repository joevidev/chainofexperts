# Communicative Mixture of Experts - Repository Structure

```
communicative-moe/
│
├── README.md                  # Project overview, setup instructions, and usage guide
├── requirements.txt           # Project dependencies
│
├── src/                       # Source code directory
│   ├── __init__.py            # Makes src a Python package
│   ├── models/                # Model implementations
│   │   ├── __init__.py        # Makes models a package
│   │   ├── expert.py          # Expert network implementation
│   │   ├── router.py          # Router network implementation
│   │   ├── traditional_moe.py # Traditional MoE implementation
│   │   └── communicative_moe.py # Communicative MoE implementation
│   │
│   ├── data/                  # Data generation and handling
│   │   ├── __init__.py        # Makes data a package
│   │   └── synthetic_data.py  # Synthetic data generation utilities
│   │
│   ├── training/              # Training utilities
│   │   ├── __init__.py        # Makes training a package
│   │   └── trainer.py         # Training loop and utilities
│   │
│   └── analysis/              # Analysis and visualization
│       ├── __init__.py        # Makes analysis a package
│       └── evaluation.py      # Model evaluation and visualization utilities
│
├── experiments/               # Experiment scripts
│   ├── __init__.py            # Makes experiments a package
│   ├── run_experiment.py      # Main experiment runner
│   └── analysis_tools.py      # Extended analysis tools
│
├── outputs/                   # Directory for experiment outputs (created at runtime)
│   ├── models/                # Saved model weights
│   ├── plots/                 # Generated plots and visualizations
│   └── results/               # Numerical results and metrics
│
└── notebooks/                 # Jupyter notebooks for exploration and result visualization
    └── experiment_analysis.ipynb # Notebook for analyzing experiment results
```

## Component Overview

### Models
- **Expert**: A feed-forward neural network that processes input data
- **Router**: Determines which experts to activate for each input
- **TraditionalMoE**: Standard mixture of experts model with weighted averaging
- **CommunicativeMoE**: Extended MoE with expert communication capabilities

### Data
- **SyntheticData**: Generates synthetic data with patterns that benefit from expert specialization

### Training
- **Trainer**: Handles model training, validation, and logging

### Analysis
- **Evaluation**: Compares model performance, analyzes expert specialization, and visualizes results

### Experiments
- **run_experiment.py**: Main script to execute experiments
- **analysis_tools.py**: Advanced analysis tools for interpreting results

This structure follows best practices for research code organization, separating core components into modular files while maintaining a clear hierarchy.