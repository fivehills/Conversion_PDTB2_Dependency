# Enhanced PDTB Dependency Parser

A BERT-based parser for converting PDTB (Penn Discourse Treebank) annotations to discourse dependency structures, implementing the approach described in "A Novel Dependency Framework for Enhancing Discourse Data Analysis".

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Examples](#examples)

## Overview

This implementation combines:
- Rule-based PDTB to dependency conversion
- BERT-based neural enhancement
- Support for both explicit and implicit relations
- Handling of asymmetric and symmetric relations
- Local dependency structure analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fivehills/Conversion_PDTB2_Dependency.git
cd Conversion_PDTB2_Dependency
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
torch>=1.9.0
transformers>=4.12.0
numpy>=1.19.0
tqdm>=4.64.0
scikit-learn>=0.24.0
```

## Project Structure

```
pdtb-dependency-parser/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── model.py           # Enhanced BERT parser implementation
│   ├── data_utils.py      # Data loading and processing
│   ├── metrics.py         # Evaluation metrics
│   └── train.py          # Training pipeline
├── data/
│   ├── raw/              # Raw PDTB annotations
│   └── processed/        # Processed data files
├── scripts/
│   ├── preprocess.py     # Data preprocessing scripts
│   └── evaluate.py       # Evaluation scripts
├── examples/
│   └── wsj_example.py    # Example usage with WSJ data
├── config/
│   └── default.yaml      # Default configuration
├── requirements.txt
└── README.md
```

## Implementation Details

### Key Components

1. **PDTB Parser** (`src/model.py`):
```python
class EnhancedPDTBDependencyParser(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_labels):
        # Initialize BERT and parsing layers
        
    def forward(self, input_ids, attention_mask, edu_indices):
        # Process input through BERT and parsing layers
        
    def convert_to_dependencies(self, pdtb_text):
        # Convert PDTB to dependencies
```

2. **Data Processing** (`src/data_utils.py`):
```python
class PDTBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Load and process PDTB data
```

3. **Training Pipeline** (`src/train.py`):
```python
def train_model(model, train_loader, dev_loader, config):
    # Training loop implementation
```

### Key Features

1. **Asymmetric Relation Handling**:
- Implementation of Table 1 rules from the paper
- Proper head/dependent determination
- Support for all PDTB relation types

2. **BERT Integration**:
- Contextual embeddings for EDUs
- Neural scoring of dependencies
- Enhanced relation classification

## Usage

### Basic Usage

1. Convert PDTB to dependencies:
```python
from src.model import EnhancedPDTBDependencyParser

# Initialize parser
parser = EnhancedPDTBDependencyParser()

# Convert PDTB annotations
with open('example.pdtb', 'r') as f:
    pdtb_text = f.read()
dependencies = parser.convert_to_dependencies(pdtb_text)
```

2. Process with BERT features:
```python
# Get enhanced dependencies
enhanced_deps = parser.process_text_with_bert(original_text, dependencies)
```

## Training

1. Prepare configuration:
```python
from src.config import ParserConfig

config = ParserConfig(
    bert_model_name="bert-base-uncased",
    batch_size=32,
    learning_rate=2e-5,
    num_epochs=30
)
```

2. Train model:
```python
from src.train import train_model

# Create dataloaders
train_loader, dev_loader = create_dataloaders(config)

# Train
train_model(model, train_loader, dev_loader, config)
```

## Dataset Preparation

1. PDTB annotation format:
```
Explicit|96..100|||||9..78|when|Contingency.Condition.Arg2-as-cond||||||79..94||||||101..158
```

2. Convert to training format:
```python
{
    "text": "original_text",
    "edu_indices": [list of EDU boundaries],
    "relations": [list of PDTB relations]
}
```

## Configuration

Edit `config/default.yaml`:
```yaml
model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 768
  num_labels: 19

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 30
  warmup_steps: 3000

data:
  max_length: 512
  train_file: "data/train.json"
  dev_file: "data/dev.json"
```

## Examples

### WSJ Example

```python
from examples.wsj_example import process_wsj_file

# Process WSJ 0618
wsj_file = "data/raw/wsj_0618.pdtb"
dependencies = process_wsj_file(wsj_file)

# Print results
for dep in dependencies:
    print(f"Head: {dep['head']} → Dependent: {dep['dependent']}")
    print(f"Relation: {dep['relation_type']}")
```

### Batch Processing

```python
from src.data_utils import process_pdtb_batch

files = ["file1.pdtb", "file2.pdtb", "file3.pdtb"]
results = process_pdtb_batch(files)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code, please cite:
```bibtex
@article{sun2024novel,
  title={A Novel Dependency Framework for Enhancing Discourse Data Analysis},
  author={Sun, Kun and Wang, Rong},
  journal={Data Intelligence},
  year={2024}
}
```

For questions and issues, please open an issue in the repository.
