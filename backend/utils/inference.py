"""
Inference utilities for the trained LegalBERT model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import json
import re
from typing import List, Dict
from collections import defaultdict
import numpy as np

class LegalBERTClassifier(nn.Module):
    """LegalBERT multi-label classifier"""
    
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

class ClauseSegmenter:
    """Segment contracts into clauses"""
    
    def __init__(self, min_length=50, max_length=1500):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(
            r'(?:^|\n)\s*(?:\d+\.\s+|\d+\.\d+\s+|\([a-z]\)\s+|'
            r'[A-Z][A-Z\s]{2,}:|Section\s+\d+|Article\s+[IVX]+)',
            re.MULTILINE | re.IGNORECASE
        )
    
    def segment(self, text):
        sections = self.pattern.split(text)
        clauses = []
        
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if len(sec) < self.min_length:
                continue
            
            if len(sec) > self.max_length:
                sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sec)
                current = ""
                for s in sents:
                    if len(current) + len(s) > self.max_length and current:
                        clauses.append({'id': len(clauses), 'text': current.strip()})
                        current = s
                    else:
                        current += " " + s
                if current:
                    clauses.append({'id': len(clauses), 'text': current.strip()})
            else:
                clauses.append({'id': len(clauses), 'text': sec})
        
        return clauses

class LegalNER:
    """Legal Named Entity Recognition"""
    
    def __init__(self, model_name="dslim/bert-base-NER"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.eval()
        except:
            self.model = None
    
    def extract_entities(self, text, max_length=512):
        if self.model is None:
            return self._rule_based_extraction(text)
        
        inputs = self.tokenizer(text, max_length=max_length, truncation=True, 
                               return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        current_entity = None
        
        for idx, (token, pred_id, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.model.config.id2label[pred_id.item()]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token.replace('##', ''),
                    'type': label[2:],
                    'start': offset[0].item(),
                    'end': offset[1].item()
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += token.replace('##', '')
                current_entity['end'] = offset[1].item()
            elif label == 'O' and current_entity:
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _rule_based_extraction(self, text):
        entities = []
        
        # Dates
        for match in re.finditer(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
            entities.append({'text': match.group(), 'type': 'DATE', 
                           'start': match.start(), 'end': match.end()})
        
        # Money
        for match in re.finditer(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text):
            entities.append({'text': match.group(), 'type': 'MONEY', 
                           'start': match.start(), 'end': match.end()})
        
        # Percentages
        for match in re.finditer(r'\d+(?:\.\d+)?\s*%', text):
            entities.append({'text': match.group(), 'type': 'PERCENT', 
                           'start': match.start(), 'end': match.end()})
        
        # Durations
        for match in re.finditer(r'\d+\s*(?:days?|weeks?|months?|years?)', text, re.IGNORECASE):
            entities.append({'text': match.group(), 'type': 'DURATION', 
                           'start': match.start(), 'end': match.end()})
        
        return entities

class ContractAnalyzer:
    """Complete contract analysis pipeline"""
    
    def __init__(self, classifier, ner_model, tokenizer, categories, device):
        self.segmenter = ClauseSegmenter()
        self.classifier = classifier
        self.ner = ner_model
        self.tokenizer = tokenizer
        self.categories = categories
        self.device = device
        self.classifier.eval()
    
    def analyze_contract(self, contract_text, classification_threshold=0.4, top_k_clauses=10):
        # Segment
        clauses = self.segmenter.segment(contract_text)
        
        # Classify and extract entities
        analyzed_clauses = []
        all_entities = []
        category_counts = defaultdict(int)
        
        for clause in clauses:
            # Classify
            inputs = self.tokenizer(clause['text'], max_length=256, padding='max_length',
                                   truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
                probs = torch.sigmoid(logits)[0]
            
            predictions = []
            for idx, prob in enumerate(probs):
                if prob > classification_threshold:
                    predictions.append({
                        'label': self.categories[idx],
                        'score': float(prob)
                    })
            
            # Extract entities
            entities = self.ner.extract_entities(clause['text'])
            
            # Calculate importance
            importance = self._calculate_importance(predictions, entities)
            
            clause_result = {
                'clause_id': clause['id'],
                'text': clause['text'],
                'predicted_labels': predictions,
                'entities': entities,
                'importance_score': importance,
                'num_labels': len(predictions),
                'num_entities': len(entities)
            }
            
            analyzed_clauses.append(clause_result)
            all_entities.extend(entities)
            
            for pred in predictions:
                category_counts[pred['label']] += 1
        
        # Rank
        analyzed_clauses.sort(key=lambda x: x['importance_score'], reverse=True)
        for rank, clause in enumerate(analyzed_clauses):
            clause['importance_rank'] = rank + 1
        
        # Summary
        summary = {
            'total_clauses': len(clauses),
            'clauses_with_labels': sum(1 for c in analyzed_clauses if c['num_labels'] > 0),
            'total_entities': len(all_entities),
            'unique_categories': len(category_counts),
            'top_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'entity_distribution': self._count_entity_types(all_entities),
            'high_importance_clauses': sum(1 for c in analyzed_clauses if c['importance_score'] > 0.7)
        }
        
        return {
            'contract_analysis': {
                'total_length': len(contract_text),
                'summary_statistics': summary,
                'all_clauses': analyzed_clauses,
                'top_important_clauses': analyzed_clauses[:top_k_clauses]
            }
        }
    
    def _calculate_importance(self, predictions, entities):
        if not predictions:
            return 0.0
        avg_confidence = np.mean([p['score'] for p in predictions])
        multi_label_bonus = min(len(predictions) / 5.0, 0.3)
        entity_bonus = min(len([e for e in entities if e['type'] in ['DATE', 'MONEY', 'PERCENT']]) * 0.1, 0.3)
        return float(min(avg_confidence + multi_label_bonus + entity_bonus, 1.0))
    
    def _count_entity_types(self, entities):
        counts = defaultdict(int)
        for entity in entities:
            counts[entity['type']] += 1
        return dict(counts)

class LegalNLPInferenceAPI:
    """Main inference API"""
    
    def __init__(self, model_path, config_path, device='cpu'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # FIXED: Use locally defined LegalBERTClassifier (no external import needed)
        self.classifier = LegalBERTClassifier(
            num_labels=self.config['num_categories'],
            dropout=self.config['dropout']
        )
        
        # FIXED: Handle checkpoint wrapper robustly
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        self.classifier.load_state_dict(model_state_dict)
        self.classifier.to(device)
        self.classifier.eval()
        
        # FIXED: Use locally defined classes (no external imports needed)
        self.ner = LegalNER()
        
        self.analyzer = ContractAnalyzer(
            classifier=self.classifier,
            ner_model=self.ner,
            tokenizer=self.tokenizer,
            categories=self.config['categories'],
            device=device
        )
    
    def analyze(self, contract_text, threshold=0.4):
        return self.analyzer.analyze_contract(contract_text, classification_threshold=threshold)
    
    def quick_classify(self, clause_text, threshold=0.5):
        inputs = self.tokenizer(clause_text, max_length=256, padding='max_length',
                               truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.sigmoid(logits)[0]
        
        predictions = []
        for idx, prob in enumerate(probs):
            if prob > threshold:
                predictions.append({
                    'label': self.config['categories'][idx],
                    'score': float(prob)
                })
        return sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    def extract_entities(self, text):
        return self.ner.extract_entities(text)
