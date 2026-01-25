# """
# DeepMindCheck ML Models Package
# Production mental health text classification
# """

# from .predictor import get_predictor, predict_text, MentalHealthPredictor

# __all__ = ['get_predictor', 'predict_text', 'MentalHealthPredictor']



"""
ML Models Integration - Baseline and DistilBERT
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
    Unified predictor that handles both baseline and DistilBERT models
    """
    
    def __init__(self):
        self.baseline_model = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
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
    
    def load_models(self):
        """Load all available models"""
        baseline_loaded = self.load_baseline_model()
        distilbert_loaded = self.load_distilbert_model()
        
        self.is_loaded = baseline_loaded or distilbert_loaded
        
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
    
    def predict(self, text, model_choice='baseline', include_probabilities=True):
        """
        Main prediction method - routes to appropriate model
        
        Args:
            text: Input text to analyze
            model_choice: 'baseline', 'advanced', or 'ensemble'
            include_probabilities: Whether to include full probability distribution
        """
        try:
            if model_choice == 'baseline':
                if self.baseline_model is None:
                    logger.warning("Baseline model not available, using DistilBERT")
                    return self.predict_distilbert(text)
                return self.predict_baseline(text)
            
            elif model_choice == 'advanced':
                if self.distilbert_model is None:
                    logger.warning("DistilBERT not available, using baseline")
                    return self.predict_baseline(text)
                return self.predict_distilbert(text)
            
            elif model_choice == 'ensemble':
                # Use both models and average predictions
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
            
            # Get baseline prediction
            if self.baseline_model is not None:
                baseline_result = self.predict_baseline(text)
                results.append(baseline_result)
            
            # Get DistilBERT prediction
            if self.distilbert_model is not None:
                distilbert_result = self.predict_distilbert(text)
                results.append(distilbert_result)
            
            if not results:
                raise Exception("No models available for ensemble")
            
            # Average probabilities
            avg_probs = {}
            for label in self.id2label.values():
                avg_probs[label] = np.mean([r['probabilities'][label] for r in results])
            
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
            'device': str(self.device),
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
        _predictor.load_models()
    
    return _predictor
