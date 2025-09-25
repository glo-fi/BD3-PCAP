# Flow Block Diffusion Model

A minimal working example of a Block Diffusion model (inspired by BD3-LMs) adapted for network flow statistics generation. This implementation currently trains a model on flow statistics where each 'block' corresponds to a single network flow, with the aim of expanding to generating more complex packet captures. Note that this code is a considerable simplification and does not implement many of the contributions of the BD3-LMs paper.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individual packages:
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
```

## Quick Start

### Run the Quick Script

The easiest way to get started is with the quick script:

```bash
python quick.py
```

This will:
- Create sample network flows with 6 features each
- Train a small BD3-LM model
- Generate new flow samples
- Evaluate the generation quality
- Create visualisation plots

## Architecture

### Flow Tokenisation

The system converts numerical flow statistics to discrete tokens using a binning strategy:

1. **Feature Binning**: Each numerical feature is discretised into bins (default: 256 bins per feature)
2. **Token Mapping**: Bin IDs become tokens, with a special mask token for diffusion
3. **Sequence Formation**: Each flow becomes a sequence of feature tokens

### Model Architecture

- **Transformer-based**: Uses multi-head attention with rotary positional embeddings
- **Block-Causal Attention**: Allows attention within flow blocks and to previous flows
- **Masked Diffusion**: Trains by randomly masking flow features and learning to reconstruct

### Flow Features

Default generated flow features include:
- `duration`
- `total_packets`
- `total_bytes`
- `packets_per_sec`
- `bytes_per_sec`
- `avg_packet_size`
- `protocol`
- `port_ratio`

These are just placeholders for this WIP implementation and will be expanded as the project grows.

## Configuration

Model and training parameters can be configured:

```python
config = {
    # Data parameters
    'n_flows': 10000,           # Number of training flows
    'n_features': 8,            # Features per flow
    'sequence_length': 8,       # Max sequence length
    'n_bins': 256,              # Bins per feature
    'batch_size': 64,           # Training batch size
    
    # Model parameters  
    'd_model': 256,             # Hidden dimension
    'n_heads': 8,               # Attention heads
    'n_layers': 6,              # Transformer layers
    'block_size': 1,            # Block size (1 = per-flow)
    
    # Training parameters
    'epochs': 100,              # Training epochs
    'learning_rate': 3e-4,      # Learning rate
    'mask_prob': 0.15,          # Masking probability
}
```

## Evaluation Metrics

The system provides comprehensive evaluation:

### Distribution Metrics
- **KS Test**
- **Earth Mover's Distance**
- **Mean and Standard Deviation Differences**
- **Distance to Real Data*

### Correlation Metrics
- **Correlation Matrix Comparison**
- **Frobenius Norm**

## File Structure

```
flow_diffusion/
├── __init__.py           # Package initialisation
├── data.py              # Dataset and tokenisation
├── model.py             # BD3-LM model architecture  
├── train.py             # Training script
└── evaluation.py        # Evaluation metrics

quick.py                  # quickrun script
generate_flows.py        # CLI generation tool
requirements.txt         # Dependencies
README.md               # This file
```

## Example Usage

### Training on Custom Data

```python
from flow_diffusion import create_flow_dataloader, FlowBD3LM, FlowTrainer
import numpy as np

# Load your flow data (CSV with numerical features)
data = np.loadtxt('my_flows.csv', delimiter=',', skiprows=1)
feature_names = ['duration', 'packets', 'bytes', 'protocol']

# Create data loaders
train_loader, val_loader, tokenizer = create_flow_dataloader(
    data=data,
    feature_names=feature_names,
    batch_size=32,
    sequence_length=len(feature_names),
    n_bins=256
)

# Create and train model
model = FlowBD3LM(vocab_size=tokenizer.vocab_size)
trainer = FlowTrainer(model, train_loader, val_loader, tokenizer, config)
trainer.train()
```

### Generating Flows

```python
from flow_diffusion import generate_flows
import torch

# Load trained model
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate new flows
generated_flows = generate_flows(
    model=model,
    tokenizer=tokenizer, 
    n_samples=1000,
    temperature=0.8
)

print(f"Generated {len(generated_flows)} flows with shape {generated_flows.shape}")
```

## Future Extensions

- **Ingest Network Streams Directly**: Implement a PCAP processing pipeline that calculates statistics before training/evaluating the model
- **Generate PCAPs**: Generate (truncated) PCAPs directly, similar to FlowDiffusion paper. It is unlikely that we will be able to generate entire network reconstrcutions, but single flows of ~1000 packets may be achievable.
- **Better Metrics**: Expand suite of metrics to include, e.g., $\alpha$-precision, $\beta$-recall, authenticity. Furthermore, some domain specific metrics will probably be needed (e.g., protocol compliance)
- **Conditional Generation**: Generate flows conditioned on protocols or network conditions
- **Advanced Tokenisation**: Learned embeddings instead of binning


## Citation

This work is partially based on the Block Diffusion paper:

```bibtex
@inproceedings{arriola2025block,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Gokaslan, Aaron Kerem and Chiu, Justin T and Yang, Zhihan and Qi, Zhixuan and Han, Jiaqi and Sahoo, Subham Sekhar and Kuleshov, Volodymyr},
  booktitle={ICLR},
  year={2025}
}
```

This work deviates significantly in many ways as a proof-of-concept. For instance, we forgoe BD3-LMs NELBO formulation and their clipped noise schedules to minimise gradient variance.

Much of this repo is also based on code from the corresponding BD3-LMs repo.