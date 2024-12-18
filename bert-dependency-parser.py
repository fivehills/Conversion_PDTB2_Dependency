
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParserConfig:
    bert_model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_labels: int = 19
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 30
    max_length: int = 512
    warmup_steps: int = 3000
    train_file: str = "data/train.json"
    dev_file: str = "data/dev.json"
    test_file: str = "data/test.json"

# data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader
import json

class PDTBDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'edu_indices': item['edu_indices'],
            'relations': item['relations']
        }

def create_dataloaders(config: ParserConfig, tokenizer) -> tuple:
    train_dataset = PDTBDataset(config.train_file, tokenizer)
    dev_dataset = PDTBDataset(config.dev_file, tokenizer)
    test_dataset = PDTBDataset(config.test_file, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    return train_loader, dev_loader, test_loader

# metrics.py
def calculate_metrics(predictions, labels):
    """Calculate UAS, LAS and other metrics"""
    uas = calculate_uas(predictions, labels)
    las = calculate_las(predictions, labels)
    return {
        'UAS': uas,
        'LAS': las
    }

def calculate_uas(predictions, labels):
    """Calculate Unlabeled Attachment Score"""
    correct = sum(1 for pred, gold in zip(predictions, labels) if pred['head'] == gold['head'])
    total = len(predictions)
    return correct / total if total > 0 else 0

def calculate_las(predictions, labels):
    """Calculate Labeled Attachment Score"""
    correct = sum(1 for pred, gold in zip(predictions, labels) 
                 if pred['head'] == gold['head'] and pred['relation_type'] == gold['relation_type'])
    total = len(predictions)
    return correct / total if total > 0 else 0

# train.py
from tqdm import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def train_model(model, train_loader, dev_loader, config):
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': config.learning_rate},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert')], 
         'lr': config.learning_rate * 10}
    ])
    
    criterion = CrossEntropyLoss()
    best_metrics = {'UAS': 0, 'LAS': 0}
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            optimizer.zero_grad()
            
            # Forward pass
            arc_scores, label_scores, _ = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['edu_indices']
            )
            
            # Calculate loss
            loss = calculate_loss(arc_scores, label_scores, batch['relations'], criterion)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on dev set
        metrics = evaluate_model(model, dev_loader)
        
        # Save best model
        if metrics['LAS'] > best_metrics['LAS']:
            best_metrics = metrics
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch {epoch}: Loss={total_loss/len(train_loader)}, UAS={metrics["UAS"]}, LAS={metrics["LAS"]}')

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    gold_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get model predictions
            arc_scores, label_scores, _ = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['edu_indices']
            )
            
            # Convert predictions to dependencies
            batch_deps = model.convert_predictions_to_dependencies(arc_scores, label_scores)
            predictions.extend(batch_deps)
            gold_labels.extend(batch['relations'])
    
    return calculate_metrics(predictions, gold_labels)

# main.py
def main():
    # Load configuration
    config = ParserConfig()
    
    # Initialize model
    model = EnhancedPDTBDependencyParser(
        bert_model_name=config.bert_model_name,
        hidden_size=config.hidden_size,
        num_labels=config.num_labels
    )
    
    # Create dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(config, model.tokenizer)
    
    # Train model
    train_model(model, train_loader, dev_loader, config)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate_model(model, test_loader)
    print(f'Test Results: UAS={test_metrics["UAS"]}, LAS={test_metrics["LAS"]}')

if __name__ == "__main__":
    main()

