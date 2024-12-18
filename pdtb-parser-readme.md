# PDTB Dependency Parser

This repository contains an implementation of a BERT-based PDTB dependency parser, based on the NISHIDA22-ARC-MOD architecture. The parser converts PDTB relations into dependency structures and parses them with state-of-the-art accuracy.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=1.9.0
- transformers>=4.5.0
- numpy>=1.19.2
- tqdm>=4.50.0

### Clone Repository

```bash
git clone https://github.com/yourusername/pdtb-dependency-parser
cd pdtb-dependency-parser
```

## Data Preparation

Your input data should be in JSON format with the following structure:

```json
{
    "documents": [
        {
            "text": "The full document text...",
            "edus": [
                {
                    "text": "EDU 1 text",
                    "start": 0,
                    "end": 10
                },
                // ... more EDUs
            ],
            "relations": [
                {
                    "type": "Purpose",
                    "subtype": "Arg2-as-goal",
                    "arg1": {"edu_index": 9},
                    "arg2": {"edu_index": 10}
                },
                // ... more relations
            ]
        }
    ]
}
```

### Data Processing Script

```python
from pdtb_parser.dataset import PDTBDataset

# Create training dataset
train_dataset = PDTBDataset(
    data_path="path/to/train.json",
    tokenizer=model.tokenizer
)
```

## Model Architecture

The parser consists of several key components:

1. **BERT Encoder**: Encodes input text into contextual representations
2. **Biaffine Attention**: Scores potential head-dependent pairs
3. **Relation Classifier**: Predicts relation types between EDUs

```python
# Initialize model
model = PDTBDependencyParser(bert_model_name="bert-base-uncased")
```

## Usage

### Basic Usage

```python
from pdtb_parser.model import PDTBDependencyParser

# Initialize model
parser = PDTBDependencyParser()

# Load trained model
parser.load_state_dict(torch.load("path/to/saved_model.pth"))

# Parse text
text = "dealers should slash stocks to between 15 and 30 days to reduce the costs of financing inventory"
dependencies = parser.parse(text)
```

### Batch Processing

```python
# Process multiple documents
texts = ["text1", "text2", "text3"]
batch_dependencies = parser.parse_batch(texts)
```

## Training

To train the parser:

```python
from pdtb_parser.trainer import train_parser

# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1)

# Train model
train_parser(
    model=model,
    train_loader=train_loader,
    dev_loader=dev_loader,
    num_epochs=30
)
```

### Training Parameters

- Learning rates:
  - BERT: 2e-5
  - Task-specific: 1e-4
- Warmup steps: 3000
- Early stopping patience: 10
- Batch size: 1

## Evaluation

```python
from pdtb_parser.evaluation import evaluate

# Evaluate on test set
test_loader = DataLoader(test_dataset, batch_size=1)
test_score = evaluate(model, test_loader)
print(f"Test score: {test_score:.4f}")
```

## Examples

### Example 1: Simple Relation Parsing

```python
# Parse a simple relation
text = "dealers should slash stocks to between 15 and 30 days to reduce the costs of financing inventory"
result = parser.parse(text)
print(result)
# Output:
# {
#     'head': 9,
#     'dependent': 10,
#     'relation': 'Purpose',
#     'distance': 1
# }
```

### Example 2: Complex Document Parsing

```python
# Parse a document with multiple relations
with open("example_doc.txt", "r") as f:
    text = f.read()

dependencies = parser.parse(text)
for dep in dependencies:
    print(f"Head: {dep['head']} -> Dependent: {dep['dependent']}")
    print(f"Relation: {dep['relation']}")
    print(f"Distance: {dep['distance']}")
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this parser in your research, please cite:

```bibtex
@article{pdtb_dependency_parser,
    title={A Novel Dependency Framework for Enhancing Discourse Data Analysis},
    author={Sun, Kun and Wang, Rong},
    journal={Data Intelligence},
    year={2024}
}
```

## Contact

For any questions or issues, please open an issue on GitHub or contact the authors.
