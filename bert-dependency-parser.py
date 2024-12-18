
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json
import numpy as np

class PDTBRelationTypes:
    """Define PDTB relation types and their properties"""
    ASYMMETRIC_RELATIONS = {
        'Condition': {'Arg2-as-cond': 'head_is_arg1', 'Arg1-as-cond': 'head_is_arg2'},
        'Purpose': {'Arg2-as-goal': 'head_is_arg1', 'Arg1-as-goal': 'head_is_arg2'},
        'Concession': {'Arg2-as-denier': 'head_is_arg1', 'Arg1-as-denier': 'head_is_arg2'},
        'Exception': {'Arg2-as-except': 'head_is_arg1', 'Arg1-as-except': 'head_is_arg2'},
        'Level-of-detail': {'Arg2-as-detail': 'head_is_arg1', 'Arg1-as-detail': 'head_is_arg2'},
        'Manner': {'Arg2-as-manner': 'head_is_arg1', 'Arg1-as-manner': 'head_is_arg2'},
        'Substitution': {'Arg2-as-subst': 'head_is_arg1', 'Arg1-as-subst': 'head_is_arg2'}
    }
    
    SYMMETRIC_RELATIONS = [
        'synchronous', 'asynchronous', 'cause',
        'contrast', 'similarity', 'conjunction',
        'disjunction'
    ]

    @classmethod
    def get_relation_index(cls, relation: str) -> int:
        """Convert relation string to index"""
        all_relations = list(cls.ASYMMETRIC_RELATIONS.keys()) + cls.SYMMETRIC_RELATIONS
        return all_relations.index(relation)

    @classmethod
    def get_relation_from_index(cls, index: int) -> str:
        """Convert index to relation string"""
        all_relations = list(cls.ASYMMETRIC_RELATIONS.keys()) + cls.SYMMETRIC_RELATIONS
        return all_relations[index]

class BiaffineAttention(nn.Module):
    """Biaffine attention for scoring head-dependent pairs"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinear = nn.Bilinear(in_features, in_features, out_features)
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        bilinear_output = self.bilinear(x1, x2)
        linear_output = self.linear(torch.cat([x1, x2], dim=-1))
        return bilinear_output + linear_output

class PDTBDependencyParser(nn.Module):
    """PDTB Dependency Parser with BERT encoder"""
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.hidden_size = 768  # BERT base hidden size
        
        # MLPs for arc prediction
        self.mlp_arc_h = nn.Sequential(
            nn.Linear(self.hidden_size, 120),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.mlp_arc_d = nn.Sequential(
            nn.Linear(self.hidden_size, 120),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Biaffine attention for arc scoring
        self.arc_biaffine = BiaffineAttention(120, 1)
        
        # Relation classifier
        num_relations = len(PDTBRelationTypes.ASYMMETRIC_RELATIONS) + \
                       len(PDTBRelationTypes.SYMMETRIC_RELATIONS)
        self.relation_classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_relations)
        )

    def forward(self, 
                text_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                edu_boundaries: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model"""
        # Get BERT encodings
        bert_output = self.bert(text_ids, attention_mask=attention_mask).last_hidden_state
        
        # Get EDU representations
        edu_reps = []
        for start, end in edu_boundaries:
            edu_rep = bert_output[:, start:end].mean(dim=1)
            edu_reps.append(edu_rep)
        edu_reps = torch.stack(edu_reps, dim=1)
        
        # Arc scoring
        arc_h = self.mlp_arc_h(edu_reps)
        arc_d = self.mlp_arc_d(edu_reps)
        arc_scores = self.arc_biaffine(arc_h, arc_d)
        
        # Relation prediction
        batch_size, seq_len = edu_reps.size(0), edu_reps.size(1)
        head_reps = edu_reps.unsqueeze(2).expand(-1, -1, seq_len, -1)
        dep_reps = edu_reps.unsqueeze(1).expand(-1, seq_len, -1, -1)
        relation_input = torch.cat([head_reps, dep_reps], dim=-1)
        relation_scores = self.relation_classifier(relation_input)
        
        return arc_scores, relation_scores

class PDTBDataset(Dataset):
    """Dataset class for PDTB parsing"""
    def __init__(self, data_path: str, tokenizer: BertTokenizer):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path: str) -> List[Dict]:
        """Load PDTB data from file"""
        with open(data_path, 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single instance"""
        instance = self.data[idx]
        
        # Tokenize text
        tokenized = self.tokenizer(
            instance['text'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        return {
            'text_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'edu_boundaries': instance['edu_boundaries'],
            'arc_labels': torch.tensor(instance['arc_labels']),
            'relation_labels': torch.tensor(instance['relation_labels'])
        }

def train_parser(model: PDTBDependencyParser,
                train_loader: DataLoader,
                dev_loader: DataLoader,
                num_epochs: int = 30):
    """Train the parser"""
    # Optimizers
    bert_optimizer = AdamW(model.bert.parameters(), lr=2e-5)
    task_optimizer = AdamW([
        {'params': model.mlp_arc_h.parameters()},
        {'params': model.mlp_arc_d.parameters()},
        {'params': model.arc_biaffine.parameters()},
        {'params': model.relation_classifier.parameters()}
    ], lr=1e-4)
    
    # Loss functions
    arc_criterion = nn.BCEWithLogitsLoss()
    relation_criterion = nn.CrossEntropyLoss()
    
    best_dev_score = 0
    patience = 10
    no_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Forward pass
            arc_scores, relation_scores = model(
                batch['text_ids'],
                batch['attention_mask'],
                batch['edu_boundaries']
            )
            
            # Calculate loss
            arc_loss = arc_criterion(arc_scores, batch['arc_labels'])
            relation_loss = relation_criterion(relation_scores.view(-1, relation_scores.size(-1)),
                                            batch['relation_labels'].view(-1))
            loss = arc_loss + relation_loss
            
            # Backward pass
            loss.backward()
            bert_optimizer.step()
            task_optimizer.step()
            bert_optimizer.zero_grad()
            task_optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Evaluate on dev set
        dev_score = evaluate(model, dev_loader)
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}, Dev Score: {dev_score}")
        
        # Early stopping
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered")
                break

def evaluate(model: PDTBDependencyParser, data_loader: DataLoader) -> float:
    """Evaluate the parser"""
    model.eval()
    correct_arcs = 0
    correct_relations = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            arc_scores, relation_scores = model(
                batch['text_ids'],
                batch['attention_mask'],
                batch['edu_boundaries']
            )
            
            # Get predictions
            arc_preds = (arc_scores > 0).float()
            relation_preds = relation_scores.argmax(dim=-1)
            
            # Calculate accuracy
            correct_arcs += (arc_preds == batch['arc_labels']).sum().item()
            correct_relations += (relation_preds == batch['relation_labels']).sum().item()
            total += batch['arc_labels'].numel()
    
    # Return unlabeled attachment score
    return correct_arcs / total

def main():
    # Initialize model and tokenizer
    model = PDTBDependencyParser()
    tokenizer = model.tokenizer
    
    # Create datasets
    train_dataset = PDTBDataset('train.json', tokenizer)
    dev_dataset = PDTBDataset('dev.json', tokenizer)
    test_dataset = PDTBDataset('test.json', tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Train model
    train_parser(model, train_loader, dev_loader)
    
    # Final evaluation
    test_score = evaluate(model, test_loader)
    print(f"Final test score: {test_score:.4f}")

if __name__ == "__main__":
    main()
