import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
import requests
from typing import List, Dict
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel



# ============================================================================
# LEGALBERT CLAUSE CLASSIFIER (UNCHANGED)
# ============================================================================

class LegalBERTClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


# ============================================================================
# CLAUSE SEGMENTER (ROBUST FOR CONTRACTS)
# ============================================================================

class ClauseSegmenter:
    def __init__(self, min_length=80, max_length=1200):
        self.min_length = min_length
        self.max_length = max_length

    def segment(self, text: str) -> List[Dict]:
        blocks = re.split(
            r'\n\s*(?:\d+\.\d+|\d+\.|\([a-z]\)|SECTION\s+\d+|ARTICLE\s+[IVX]+)\s*',
            text,
            flags=re.IGNORECASE
        )

        clauses = []
        for block in blocks:
            block = block.strip()
            if self.min_length <= len(block) <= self.max_length:
                clauses.append({
                    "id": len(clauses),
                    "text": block
                })

        # fallback if segmentation fails
        if len(clauses) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) > self.max_length:
                    if len(chunk) >= self.min_length:
                        clauses.append({"id": len(clauses), "text": chunk})
                    chunk = s
                else:
                    chunk += " " + s
            if len(chunk) >= self.min_length:
                clauses.append({"id": len(clauses), "text": chunk})

        return clauses


# ============================================================================
# GENAI NER USING PHI-3 (NO LOCAL MODEL DOWNLOAD)
# ============================================================================

# ============================================================================
# GENAI NER USING OLLAMA (PHI-3 MINI)
# ============================================================================

class LegalNER:
    """
    GenAI-based Legal NER using Phi-3 Mini via Ollama (CPU-friendly)
    """

    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"
        print(f"🤖 Using Ollama GenAI NER → {model_name}")

    def extract_entities(self, text: str) -> List[Dict]:
        prompt = f"""
You are a legal NLP system.

From the clause below, extract named entities relevant to legal contracts.

Return ONLY valid JSON in this format:
[
  {{
    "text": "...",
    "type": "ORG | PERSON | DATE | MONEY | LAW | GPE | DURATION | PARTY_ROLE | DEFINED_TERM"
  }}
]

Rules:
- PARTY_ROLE examples: Affiliate, Licensor, Licensee, Company
- DEFINED_TERM = capitalized contractual terms
- Do NOT hallucinate entities
- If none found, return []

Clause:
\"\"\"{text}\"\"\"
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            raw = response.json()["response"].strip()

            # Extract JSON safely
            json_start = raw.find("[")
            json_end = raw.rfind("]") + 1
            if json_start == -1 or json_end == -1:
                return []

            entities = json.loads(raw[json_start:json_end])

            # Minimal validation
            clean = []
            for e in entities:
                if "text" in e and "type" in e and len(e["text"]) > 2:
                    clean.append(e)

            return clean

        except Exception as e:
            print(f"⚠️ NER fallback (error): {e}")
            return []


# ============================================================================
# CONTRACT ANALYZER (CLASSIFICATION + GENAI NER)
# ============================================================================

class ContractAnalyzer:
    def __init__(self, classifier, ner, tokenizer, categories, device):
        self.segmenter = ClauseSegmenter()
        self.classifier = classifier
        self.ner = ner
        self.tokenizer = tokenizer
        self.categories = categories
        self.device = device
        self.classifier.eval()

    def analyze_contract(self, contract_text, classification_threshold=0.4, top_k_clauses=10):
        clauses = self.segmenter.segment(contract_text)

        analyzed = []
        all_entities = []
        category_counts = defaultdict(int)

        for clause in clauses:
            inputs = self.tokenizer(
                clause["text"],
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.classifier(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.sigmoid(logits)[0]

            preds = []
            for i, p in enumerate(probs):
                if p > classification_threshold:
                    preds.append({
                        "label": self.categories[i],
                        "score": float(p)
                    })
                    category_counts[self.categories[i]] += 1

            entities = self.ner.extract_entities(clause["text"])
            all_entities.extend(entities)

            importance = self._importance(preds, entities)

            analyzed.append({
                "clause_id": clause["id"],
                "text": clause["text"],
                "predicted_labels": preds,
                "entities": entities,
                "importance_score": importance,
                "num_labels": len(preds),
                "num_entities": len(entities)
            })

        analyzed.sort(key=lambda x: x["importance_score"], reverse=True)

        summary = {
            "total_clauses": len(clauses),
            "clauses_with_labels": sum(c["num_labels"] > 0 for c in analyzed),
            "total_entities": len({(e["text"], e["type"]) for e in all_entities}),
            "unique_categories": len(category_counts),
            "top_categories": sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }

        return {
            "contract_analysis": {
                "summary_statistics": summary,
                "top_important_clauses": analyzed[:top_k_clauses],
                "all_clauses": analyzed
            }
        }

    def _importance(self, preds, entities):
        if not preds:
            return 0.0
        score = np.mean([p["score"] for p in preds])
        score += min(len(entities) * 0.05, 0.3)
        return min(score, 1.0)


# ============================================================================
# INFERENCE API (USED BY FLASK)
# ============================================================================

class LegalNLPInferenceAPI:
    def __init__(self, model_path, config_path, device="cpu"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.classifier = LegalBERTClassifier(
            num_labels=self.config["num_categories"],
            dropout=self.config["dropout"]
        )

        checkpoint = torch.load(model_path, map_location=device)
        self.classifier.load_state_dict(
            checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        )
        self.classifier.to(device).eval()

        self.ner = LegalNER()

        self.analyzer = ContractAnalyzer(
            classifier=self.classifier,
            ner=self.ner,
            tokenizer=self.tokenizer,
            categories=self.config["categories"],
            device=device
        )

    def analyze(self, text, threshold=0.4):
        return self.analyzer.analyze_contract(text, classification_threshold=threshold)
