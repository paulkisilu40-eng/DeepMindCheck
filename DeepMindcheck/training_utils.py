"""
Training Utilities - Export Feedback Data
"""
import csv
import os
from django.conf import settings
from core.models import UserFeedback
from django.utils import timezone
import logging

logger = logging.getLogger('deepmindcheck')

def export_feedback_for_training(output_file=None):
    """
    Export user feedback data for model retraining.
    Exports csv with: text, original_prediction, correct_label, rating
    """
    if output_file is None:
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        output_dir = settings.BASE_DIR / 'training_data'
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / f'feedback_data_{timestamp}.csv'
    
    try:
        # Get feedback where user provided a correction or high/low rating
        feedbacks = UserFeedback.objects.select_related('analysis').all()
        
        count = 0
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text', 'original_prediction', 'correct_label', 'rating', 'feedback_text', 'created_at']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for feedback in feedbacks:
                # Determine "correct" label:
                # 1. User specified correct_label
                # 2. Or implied correct if rating is high (5) -> predicted was correct
                
                label = feedback.correct_label
                if not label and feedback.rating == 5:
                    label = feedback.analysis.prediction
                
                # Only export if we have a target label
                if label:
                    writer.writerow({
                        'text': feedback.analysis.text_input,
                        'original_prediction': feedback.analysis.prediction,
                        'correct_label': label,
                        'rating': feedback.rating,
                        'feedback_text': feedback.feedback_text,
                        'created_at': feedback.created_at.isoformat()
                    })
                    count += 1
        
        logger.info(f"Exported {count} training examples to {output_file}")
        return output_file, count
        
    except Exception as e:
        logger.error(f"Failed to export training data: {e}")
        return None, 0
