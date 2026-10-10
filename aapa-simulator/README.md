# AAPA: Archetype-Aware Predictive Autoscaler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation of AAPA (Archetype-Aware Predictive Autoscaler), a novel autoscaling system for serverless workloads on Kubernetes that leverages workload classification and uncertainty quantification.

## 🎯 Key Features

- **Workload Classification**: Automatically classifies serverless workloads into 4 archetypes (SPIKE, PERIODIC, RAMP, STATIONARY) with 99.8% accuracy
- **Weak Supervision**: Uses programmatic labeling functions to automatically label 300K+ time windows
- **Uncertainty-Aware Scaling**: Incorporates prediction confidence to adjust scaling aggressiveness
- **Differentiated Strategies**: Applies archetype-specific scaling policies for optimal performance
- **Public Dataset**: Includes the Kubernetes Serverless Workload Dataset (KSWD) with 10,000+ real-world traces

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure Functions 2019 dataset (download link below)
- 16GB+ RAM recommended for full experiments

### Installation

```bash
# Clone the repository
git clone https://github.com/GuilinDev/aapa-simulator.git
cd aapa-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the Azure Functions 2019 dataset:
```bash
# Create data directory
mkdir -p data/raw

# Download from Kaggle (requires Kaggle API credentials)
kaggle datasets download -d azurepublicdataset/azurefunctions-dataset2019 -p data/raw/
unzip data/raw/azurefunctions-dataset2019.zip -d data/raw/
```

Alternatively, download manually from: https://www.kaggle.com/datasets/azurepublicdataset/azurefunctions-dataset2019

### Run Experiments

```bash
# Phase 1: Workload classification
python examples/run_classification.py

# Phase 2: Autoscaling simulation
python examples/run_simulation.py

# Generate visualizations
python examples/create_visualizations.py
```

## 📁 Project Structure

```
aapa-simulator/
├── src/
│   ├── core/              # Core autoscaling logic
│   │   ├── autoscaler.py  # Main AAPA autoscaler
│   │   ├── strategies.py  # Archetype-specific strategies
│   │   └── simulator.py   # Kubernetes simulator
│   ├── data/              # Data processing
│   │   ├── loader.py      # Azure dataset loader
│   │   ├── features.py    # Feature extraction
│   │   └── windowing.py   # Sliding window processor
│   ├── models/            # ML models
│   │   ├── classifier.py  # LightGBM classifier
│   │   ├── labeling.py    # Weak supervision
│   │   └── uncertainty.py # Uncertainty quantification
│   ├── utils/             # Utilities
│   │   └── metrics.py     # Evaluation metrics
│   └── visualization/     # Plotting functions
├── examples/              # Example scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── data/                  # Data directory
│   ├── raw/              # Raw Azure traces
│   └── processed/        # Processed features
├── dataset/              # Kubernetes Serverless Workload Dataset (KSWD)
│   ├── workloads/        # Individual workload traces
│   ├── archetypes/       # Workload groupings by type
│   ├── examples/         # Usage examples
│   └── tools/            # Dataset generation tools
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT license
└── README.md            # This file
```

## 🔬 Reproducing Paper Results

To reproduce the results from our paper:

1. **Prepare the data**:
   ```bash
   python examples/prepare_data.py --days 14
   ```

2. **Train the classifier**:
   ```bash
   python examples/train_classifier.py --output models/lightgbm_model.pkl
   ```

3. **Run full evaluation**:
   ```bash
   python examples/reproduce_paper_results.py
   ```

This will generate all figures and tables from the paper in the `results/` directory.

## 📊 Main Results

AAPA achieves:
- **50% reduction** in SLO violations for spike workloads
- **40% improvement** in response times
- **99.8% accuracy** in workload classification
- **2-8× resource overhead** (fundamental tradeoff)

## 🛠️ Configuration

Key parameters can be adjusted in `config/default.yaml`:

```yaml
# Autoscaling parameters
scaling:
  spike:
    target_cpu: 0.3
    cooldown: 20
    min_replicas: 2
  periodic:
    target_cpu: 0.75
    cooldown: 3
    min_replicas: 1
    
# Simulation parameters
simulation:
  pod_startup_time: 30  # seconds
  metric_interval: 60   # seconds
```

## 📈 Workload Archetypes

| Archetype | Characteristics | Scaling Strategy |
|-----------|----------------|------------------|
| SPIKE | Sudden bursts | Warm pools + aggressive scaling |
| PERIODIC | Regular patterns | Predictive pre-scaling |
| RAMP | Gradual changes | Trend following |
| STATIONARY | Stable + noise | Conservative with long cooldowns |

## 📦 Kubernetes Serverless Workload Dataset (KSWD)

This repository includes a comprehensive dataset of real-world serverless workload traces formatted for Kubernetes research:

- **10,000+ workload traces** spanning 14 days each
- **Archetype labels** with confidence scores
- **Resource requirements** (CPU/memory) and SLO targets
- **Ready-to-use format** for Kubernetes HPA configurations

See the [dataset documentation](dataset/README.md) for detailed information and usage examples.

### Quick Dataset Usage

```bash
# Generate the dataset from Azure traces
python dataset/tools/azure_to_k8s_converter.py \
  --azure-path /path/to/azure/data \
  --output-path dataset/ \
  --max-workloads 100

# Load and visualize workloads
python dataset/examples/load_workload.py
python dataset/examples/visualize_traces.py

# Generate Kubernetes HPA configs
python dataset/examples/generate_k8s_hpa.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2025aapa,
  title={Archetype-Aware Predictive Autoscaling with Uncertainty Quantification for Serverless Workloads on Kubernetes},
  author={Zhang, Guilin and others},
  booktitle={IEEE High Performance Extreme Computing Conference (HPEC)},
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
