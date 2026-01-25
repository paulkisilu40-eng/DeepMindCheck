"""
ML Models Integration - Baseline and BERT with Lazy Loading
"""
import os
import logging
import numpy as np
from pathlib import Path
from django.conf import settings

logger = logging.getLogger('deepmindcheck')

class MLPredictor:
    """
    Unified predictor that handles both baseline and BERT models
    Uses lazy loading to avoid startup timeouts
    """
    
    def __init__(self):
        self.baseline_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = None
        self.is_loaded = False
        
        # Lazy imports placeholders
        self.torch = None
        self.AutoTokenizer = None
        self.AutoModelForSequenceClassification = None
        
        # Label mapping
        self.id2label = {0: 'neutral', 1: 'depression', 2: 'anxiety'}
        self.label2id = {'neutral': 0, 'depression': 1, 'anxiety': 2}
        
        logger.info("MLPredictor initialized (models will load on first use)")
    
    def _lazy_import_torch(self):
        """Lazy import torch and transformers only when needed"""
        if self.torch is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Torch loaded, using device: {self.device}")
    
    def load_baseline_model(self):
        """Load your baseline model from ml_models folder"""
        try:
            import pickle
            model_path = Path(settings.BASE_DIR) / 'ml_models' / 'saved_models'
            
            baseline_path = model_path / 'best_model.pkl'
            vectorizer_path = model_path / 'best_vectorizer.pkl'
            
            if baseline_path.exists() and vectorizer_path.exists():
                with open(baseline_path, 'rb') as f:
                    self.baseline_model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.baseline_vectorizer = pickle.load(f)
                logger.info("✓ Baseline model loaded successfully")
                return True
            else:
                logger.warning(f"Baseline model files not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            return False
    
    def load_bert_model(self):
        """Load BERT model from Hugging Face"""
        try:
            # Lazy load torch/transformers
            self._lazy_import_torch()
            
            model_name = "AladeenPaul/base-bert"
            
            logger.info(f"Loading BERT model: {model_name}")
            
            self.bert_tokenizer = self.AutoTokenizer.from_pretrained(model_name)
            self.bert_model = self.AutoModelForSequenceClassification.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            logger.info("✓ BERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            return False
    
    def load_models(self):
        """Load all available models"""
        baseline_loaded = self.load_baseline_model()
        bert_loaded = self.load_bert_model()
        
        self.is_loaded = baseline_loaded or bert_loaded
        
        if not self.is_loaded:
            logger.error("❌ No models could be loaded!")
        else:
            logger.info("✓ ML models initialized")
        
        return self.is_loaded
    
    def predict_baseline(self, text):
        """Make prediction using baseline model"""
        try:
            if self.baseline_model is None:
                self.load_baseline_model()
                if self.baseline_model is None:
                    raise Exception("Baseline model not loaded")
            
            features = self.baseline_vectorizer.transform([text])
            prediction = self.baseline_model.predict(features)[0]
            probabilities = self.baseline_model.predict_proba(features)[0]
            
            prediction_label = self.id2label.get(prediction, 'neutral')
            confidence = float(np.max(probabilities))
            
            probs_dict = {
                self.id2label[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': probs_dict,
                'model_name': 'Baseline Model'
            }
            
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            raise
    
    def predict_bert(self, text):
        """Make prediction using BERT model"""
        try:
            if self.bert_model is None:
                self.load_bert_model()
                if self.bert_model is None:
                    raise Exception("BERT model not loaded")
            
            inputs = self.bert_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probabilities = self.torch.nn.functional.softmax(logits, dim=-1)
            
            prediction_id = self.torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][prediction_id])
            prediction_label = self.id2label.get(prediction_id, 'neutral')
            
            probs_dict = {
                self.id2label[i]: float(probabilities[0][i]) 
                for i in range(len(self.id2label))
            }
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': probs_dict,
                'model_name': 'BERT Advanced'
            }
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            raise
    
    def predict(self, text, model_choice='baseline', include_probabilities=True):
        """Main prediction method"""
        try:
            if model_choice == 'baseline':
                return self.predict_baseline(text)
            elif model_choice == 'advanced':
                return self.predict_bert(text)
            elif model_choice == 'ensemble':
                return self.predict_ensemble(text)
            else:
                logger.warning(f"Unknown model choice: {model_choice}, using baseline")
                return self.predict_baseline(text)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_ensemble(self, text):
        """Make prediction using ensemble of both models"""
        try:
            results = []
            
            if self.baseline_model is not None or self.load_baseline_model():
                results.append(self.predict_baseline(text))
            
            if self.bert_model is not None or self.load_bert_model():
                results.append(self.predict_bert(text))
            
            if not results:
                raise Exception("No models available for ensemble")
            
            avg_probs = {}
            for label in self.id2label.values():
                avg_probs[label] = np.mean([r['probabilities'][label] for r in results])
            
            prediction_label = max(avg_probs, key=avg_probs.get)
            confidence = avg_probs[prediction_label]
            
            return {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in avg_probs.items()},
                'model_name': 'Ensemble Model'
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'baseline_loaded': self.baseline_model is not None,
            'bert_loaded': self.bert_model is not None,
            'device': str(self.device) if self.device else 'Not loaded',
            'available_models': ['baseline', 'advanced', 'ensemble'],
            'labels': list(self.id2label.values())
        }


# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global _predictor
    
    if _predictor is None:
        _predictor = MLPredictor()
        # Don't load models here - let them load on first prediction
    
    return _predictor