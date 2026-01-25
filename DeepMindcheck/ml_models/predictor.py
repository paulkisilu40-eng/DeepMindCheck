# """
# DeepMindCheck ML Predictor Module
# Loads and uses trained mental health classification models
# """

# import pickle
# import json
# import numpy as np
# from pathlib import Path
# import logging

# logger = logging.getLogger('deepmindcheck')

# class MentalHealthPredictor:
#     """
#     Production-ready mental health text classifier
#     Uses trained gradient boosting model with character-level TF-IDF
#     """
    
#     def __init__(self, models_dir=None):
#         """Initialize predictor and load models"""
#         if models_dir is None:
#             # Default to ml_models/saved_models in project root
#             self.models_dir = Path(__file__).parent / 'saved_models'
#         else:
#             self.models_dir = Path(models_dir)
        
#         self.model = None
#         self.vectorizer = None
#         self.label_encoder = None
#         self.deployment_info = None
#         self.is_loaded = False
        
#         # Load models on initialization
#         self.load_models()
    
#     def load_models(self):
#         """Load all required models and configuration"""
#         try:
#             logger.info(f"Loading models from {self.models_dir}")
            
#             # Load deployment info
#             deployment_path = self.models_dir / 'deployment_info.json'
#             if deployment_path.exists():
#                 with open(deployment_path, 'r') as f:
#                     self.deployment_info = json.load(f)
#                 logger.info(f"Loaded deployment info: {self.deployment_info.get('model_name')}")
            
#             # Load model
#             model_path = self.models_dir / 'best_model.pkl'
#             if not model_path.exists():
#                 raise FileNotFoundError(f"Model file not found at {model_path}")
            
#             with open(model_path, 'rb') as f:
#                 self.model = pickle.load(f)
#             logger.info("Model loaded successfully")
            
#             # Load vectorizer
#             vectorizer_path = self.models_dir / 'best_vectorizer.pkl'
#             if not vectorizer_path.exists():
#                 raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
            
#             with open(vectorizer_path, 'rb') as f:
#                 self.vectorizer = pickle.load(f)
#             logger.info("Vectorizer loaded successfully")
            
#             # Load label encoder
#             label_encoder_path = self.models_dir / 'label_encoder.pkl'
#             if not label_encoder_path.exists():
#                 raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
            
#             with open(label_encoder_path, 'rb') as f:
#                 self.label_encoder = pickle.load(f)
#             logger.info(f"Label encoder loaded. Classes: {self.label_encoder.classes_}")
            
#             self.is_loaded = True
#             logger.info("All models loaded successfully!")
            
#             # Log performance metrics if available
#             if self.deployment_info and 'performance' in self.deployment_info:
#                 perf = self.deployment_info['performance']
#                 logger.info(f"Model performance - F1: {perf.get('f1_score', 0):.3f}, "
#                           f"Accuracy: {perf.get('accuracy', 0):.3f}")
            
#         except Exception as e:
#             logger.error(f"Failed to load models: {e}")
#             self.is_loaded = False
#             raise
    
#     def predict(self, text, include_probabilities=True):
#         """
#         Make a prediction for given text
        
#         Args:
#             text (str): Input text to classify
#             include_probabilities (bool): Include probability distribution
            
#         Returns:
#             dict: Prediction results with label, confidence, and optionally probabilities
#         """
#         if not self.is_loaded:
#             raise RuntimeError("Models not loaded. Cannot make predictions.")
        
#         # Validate input
#         if not text or not isinstance(text, str):
#             raise ValueError("Input must be a non-empty string")
        
#         text = text.strip()
#         if len(text) < 10:
#             raise ValueError("Text must be at least 10 characters long")
        
#         try:
#             # Vectorize input text
#             text_vectorized = self.vectorizer.transform([text])
            
#             # Make prediction
#             prediction_idx = self.model.predict(text_vectorized)[0]
#             predicted_label = self.label_encoder.inverse_transform([prediction_idx])[0]
            
#             # Get probabilities
#             probabilities = self.model.predict_proba(text_vectorized)[0]
#             confidence = float(max(probabilities))
            
#             # Build result
#             result = {
#                 'prediction': predicted_label,
#                 'confidence': confidence,
#                 'model_name': self.deployment_info.get('model_name', 'gradient_boosting_tfidf_chars'),
#                 'model_version': self.deployment_info.get('model_version', '1.0')
#             }
            
#             # Add full probability distribution if requested
#             if include_probabilities:
#                 result['probabilities'] = {
#                     self.label_encoder.inverse_transform([i])[0]: float(prob)
#                     for i, prob in enumerate(probabilities)
#                 }
            
#             logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
#             return result
            
#         except Exception as e:
#             logger.error(f"Prediction error: {e}")
#             raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
#     def predict_batch(self, texts):
#         """
#         Make predictions for multiple texts
        
#         Args:
#             texts (list): List of text strings
            
#         Returns:
#             list: List of prediction dictionaries
#         """
#         if not self.is_loaded:
#             raise RuntimeError("Models not loaded")
        
#         results = []
#         for text in texts:
#             try:
#                 result = self.predict(text)
#                 results.append(result)
#             except Exception as e:
#                 logger.warning(f"Failed to predict for text: {str(e)[:100]}")
#                 results.append({
#                     'error': str(e),
#                     'prediction': None,
#                     'confidence': 0.0
#                 })
        
#         return results
    
#     def get_model_info(self):
#         """Get information about the loaded model"""
#         if not self.is_loaded:
#             return {
#                 'status': 'not_loaded',
#                 'error': 'Models not loaded'
#             }
        
#         info = {
#             'status': 'loaded',
#             'model_name': self.deployment_info.get('model_name', 'Unknown'),
#             'model_type': self.deployment_info.get('model_type', 'baseline'),
#             'classes': list(self.label_encoder.classes_),
#             'n_classes': len(self.label_encoder.classes_),
#             'timestamp': self.deployment_info.get('timestamp', 'Unknown')
#         }
        
#         # Add performance metrics if available
#         if self.deployment_info and 'performance' in self.deployment_info:
#             info['performance'] = self.deployment_info['performance']
        
#         return info
    
#     def validate_prediction(self, prediction_result):
#         """
#         Validate that prediction result contains required fields
        
#         Args:
#             prediction_result (dict): Result from predict()
            
#         Returns:
#             bool: True if valid, False otherwise
#         """
#         required_fields = ['prediction', 'confidence']
#         return all(field in prediction_result for field in required_fields)
    
#     def get_confidence_level(self, confidence):
#         """
#         Categorize confidence level
        
#         Args:
#             confidence (float): Confidence score (0-1)
            
#         Returns:
#             str: 'high', 'medium', or 'low'
#         """
#         if confidence >= 0.8:
#             return 'high'
#         elif confidence >= 0.6:
#             return 'medium'
#         else:
#             return 'low'


# # Create singleton instance
# _predictor_instance = None

# def get_predictor(models_dir=None):
#     """
#     Get or create predictor singleton
    
#     Args:
#         models_dir (str): Optional custom models directory
        
#     Returns:
#         MentalHealthPredictor: Singleton predictor instance
#     """
#     global _predictor_instance
    
#     if _predictor_instance is None:
#         _predictor_instance = MentalHealthPredictor(models_dir)
    
#     return _predictor_instance


# # Convenience function for quick predictions
# def predict_text(text):
#     """
#     Quick prediction function
    
#     Args:
#         text (str): Text to classify
        
#     Returns:
#         dict: Prediction result
#     """
#     predictor = get_predictor()
#     return predictor.predict(text)


"""
ML Models Integration - Baseline and DistilBERT
Fixed version with proper model routing
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
        self.baseline_vectorizer = None
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
            model_path = Path(settings.BASE_DIR) / 'ml_models' / 'saved_models'
            
            # Try both possible paths
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
            
            logger.info(f"Loading DistilBERT model from HuggingFace: {model_name}")
            
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            logger.info("✓ DistilBERT model loaded successfully from HuggingFace")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model from HuggingFace: {e}")
            return False
    
    def load_models(self):
        """Load all available models"""
        baseline_loaded = self.load_baseline_model()
        distilbert_loaded = self.load_distilbert_model()
        
        self.is_loaded = baseline_loaded or distilbert_loaded
        
        if not self.is_loaded:
            logger.error("❌ No models could be loaded!")
        else:
            models_status = []
            if baseline_loaded:
                models_status.append("Baseline ✓")
            if distilbert_loaded:
                models_status.append("DistilBERT ✓")
            logger.info(f"✓ ML models initialized: {', '.join(models_status)}")
        
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
                'model_name': 'Baseline Model (Logistic Regression)'
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
                'model_name': 'DistilBERT Advanced (Transformer)'
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
            logger.info(f"Prediction requested with model_choice: {model_choice}")
            
            if model_choice == 'baseline':
                if self.baseline_model is None:
                    logger.warning("Baseline model not available, falling back to DistilBERT")
                    return self.predict_distilbert(text)
                return self.predict_baseline(text)
            
            elif model_choice == 'advanced':
                if self.distilbert_model is None:
                    logger.warning("DistilBERT not available, falling back to baseline")
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
                logger.info(f"Baseline: {baseline_result['prediction']} ({baseline_result['confidence']:.3f})")
            
            # Get DistilBERT prediction
            if self.distilbert_model is not None:
                distilbert_result = self.predict_distilbert(text)
                results.append(distilbert_result)
                logger.info(f"DistilBERT: {distilbert_result['prediction']} ({distilbert_result['confidence']:.3f})")
            
            if not results:
                raise Exception("No models available for ensemble")
            
            # Average probabilities
            avg_probs = {}
            for label in self.id2label.values():
                avg_probs[label] = np.mean([r['probabilities'][label] for r in results])
            
            # Get final prediction
            prediction_label = max(avg_probs, key=avg_probs.get)
            confidence = avg_probs[prediction_label]
            
            logger.info(f"Ensemble final: {prediction_label} ({confidence:.3f})")
            
            return {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in avg_probs.items()},
                'model_name': f'Ensemble Model ({len(results)} models)'
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
