"""
ML Models Integration - Baseline, DistilBERT, and BERT
"""
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
from django.conf import settings

logger = logging.getLogger('deepmindcheck')

class MLPredictor:
    """
    Unified predictor that handles baseline, DistilBERT, and BERT models
    """
    
    def __init__(self):
        self.baseline_model = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.bert_model = None
        self.bert_tokenizer = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
        # Label mapping
        self.id2label = {0: 'neutral', 1: 'depression', 2: 'anxiety'}
        self.label2id = {'neutral': 0, 'depression': 1, 'anxiety': 2}
        
        logger.info(f"Using device: {self.device}")
    
    def load_baseline_model(self):
        """Load your baseline model from ml_models folder"""
        try:
            import pickle
            model_path = Path(settings.BASE_DIR) / 'ml_models'
            
            # Adjust these filenames to match your actual files
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
    
    def load_distilbert_model(self):
        """Load DistilBERT model from Hugging Face"""
        try:
            model_name = "AladeenPaul/distil_bert"
            
            logger.info(f"Loading DistilBERT model: {model_name}")
            
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            logger.info("✓ DistilBERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            return False

    def load_bert_model(self):
        """Load BERT Base Sentiment model from Hugging Face"""
        try:
            model_name = "AladeenPaul/bert-base-sentiment"
            
            logger.info(f"Loading BERT model: {model_name}")
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
        distilbert_loaded = self.load_distilbert_model()
        bert_loaded = self.load_bert_model()
        
        self.is_loaded = baseline_loaded or distilbert_loaded or bert_loaded
        
        if not self.is_loaded:
            logger.error("❌ No models could be loaded!")
        else:
            logger.info("✓ ML models initialized")
        
        return self.is_loaded
    
    def predict_baseline(self, text):
        """Make prediction using baseline model"""
        try:
            if self.baseline_model is None:
                raise Exception("Baseline model not loaded")
            
            # Transform text
            features = self.baseline_vectorizer.transform([text])
            
            # Get prediction and probabilities
            prediction = self.baseline_model.predict(features)[0]
            probabilities = self.baseline_model.predict_proba(features)[0]
            
            # Get label
            prediction_label = self.id2label.get(prediction, 'neutral')
            confidence = float(np.max(probabilities))
            
            # Create probabilities dict
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
    
    def predict_distilbert(self, text):
        """Make prediction using DistilBERT model"""
        try:
            if self.distilbert_model is None or self.distilbert_tokenizer is None:
                raise Exception("DistilBERT model not loaded")
            
            # Tokenize
            inputs = self.distilbert_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            prediction_id = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][prediction_id])
            
            # Get label
            prediction_label = self.id2label.get(prediction_id, 'neutral')
            
            # Create probabilities dict
            probs_dict = {
                self.id2label[i]: float(probabilities[0][i]) 
                for i in range(len(self.id2label))
            }
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': probs_dict,
                'model_name': 'DistilBERT Advanced'
            }
            
        except Exception as e:
            logger.error(f"DistilBERT prediction failed: {e}")
            raise

    def predict_bert(self, text):
        """Make prediction using BERT Base Sentiment model"""
        try:
            if self.bert_model is None or self.bert_tokenizer is None:
                raise Exception("BERT model not loaded")
            
            # Tokenize
            inputs = self.bert_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            prediction_id = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][prediction_id])
            
            # Get label
            # Assuming same label mapping for ease, otherwise map accordingly
            prediction_label = self.id2label.get(prediction_id, 'neutral')
            
            # Create probabilities dict
            probs_dict = {
                self.id2label[i]: float(probabilities[0][i]) 
                for i in range(len(self.id2label))
            }
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': probs_dict,
                'model_name': 'BERT Sentiment (Advanced)'
            }
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            raise
    
    def predict(self, text, model_choice='baseline', include_probabilities=True):
        """
        Main prediction method - routes to appropriate model
        
        Args:
            text: Input text to analyze
            model_choice: 'baseline', 'advanced', 'embedded', or 'ensemble'
            include_probabilities: Whether to include full probability distribution
        """
        try:
            if model_choice == 'baseline':
                if self.baseline_model is None:
                    # Fallback chain: Baseline -> DistilBERT -> BERT
                    if self.distilbert_model is not None:
                        logger.warning("Baseline model not available, using DistilBERT")
                        return self.predict_distilbert(text)
                    elif self.bert_model is not None:
                         logger.warning("Baseline model not available, using BERT")
                         return self.predict_bert(text)
                return self.predict_baseline(text)
            
            elif model_choice == 'advanced':
                # Use BERT as the primary advanced model as user requested
                if self.bert_model is not None:
                    return self.predict_bert(text)
                # Fallback to DistilBERT
                elif self.distilbert_model is not None:
                    logger.warning("BERT not available, using DistilBERT")
                    return self.predict_distilbert(text)
                else:
                    return self.predict_baseline(text)

            elif model_choice == 'embedded':
                 # Explicitly BERT
                if self.bert_model is not None:
                    return self.predict_bert(text)
                else:
                    logger.warning("BERT (embedded) not available, falling back")
                    return self.predict(text, model_choice='advanced')
            
            elif model_choice == 'ensemble':
                # Use all available models and average predictions
                return self.predict_ensemble(text)
            
            else:
                logger.warning(f"Unknown model choice: {model_choice}, using baseline")
                return self.predict_baseline(text)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_ensemble(self, text):
        """Make prediction using ensemble of available models"""
        try:
            results = []
            
            # Get baseline prediction
            if self.baseline_model is not None:
                try:
                    results.append(self.predict_baseline(text))
                except: pass
            
            # Get DistilBERT prediction
            if self.distilbert_model is not None:
                try:
                    results.append(self.predict_distilbert(text))
                except: pass

            # Get BERT prediction
            if self.bert_model is not None:
                try:
                    results.append(self.predict_bert(text))
                except: pass
            
            if not results:
                raise Exception("No models available for ensemble")
            
            # Average probabilities
            avg_probs = {}
            for label in self.id2label.values():
                # Only average if the key exists (handling potential label mismatch if any)
                probs_list = [r['probabilities'].get(label, 0.0) for r in results]
                avg_probs[label] = np.mean(probs_list)
            
            # Get final prediction
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
            'distilbert_loaded': self.distilbert_model is not None,
            'bert_loaded': self.bert_model is not None,
            'device': str(self.device),
            'available_models': ['baseline', 'advanced', 'embedded', 'ensemble'],
            'labels': list(self.id2label.values())
        }


# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global _predictor
    
    if _predictor is None:
        _predictor = MLPredictor()
        _predictor.load_models()
    
    return _predictor
