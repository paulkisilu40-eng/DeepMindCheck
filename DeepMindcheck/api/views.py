



"""
Fixed API Views - Now properly passes model_choice to predictor
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from core.models import TextAnalysis, UserFeedback, SystemMetrics
from django.db.models import Count, Avg, Q
from django.utils import timezone
import time
import logging
import uuid

# Import our ML predictor
from ml_models import get_predictor

logger = logging.getLogger('deepmindcheck')

@api_view(['POST'])
@permission_classes([AllowAny])
def analyze_text(request):
    """
    Real ML-powered text analysis endpoint with proper model routing
    """
    try:
        # Get request data
        text = request.data.get('text', '').strip()
        model_choice = request.data.get('model', 'baseline')
        include_explanation = request.data.get('explain', False)
        
        logger.info(f"Analysis request - Model: {model_choice}, Text length: {len(text)}")
        
        # Validation
        if not text:
            return Response({
                'error': 'Text input is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(text) < 10:
            return Response({
                'error': 'Text must be at least 10 characters long'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(text) > 2000:
            return Response({
                'error': 'Text must be less than 2000 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Start timing
        start_time = time.time()
        
        # Get predictor and make prediction with model_choice parameter
        try:
            predictor = get_predictor()
            
            # 🔥 FIX: Pass model_choice to the predictor!
            ml_result = predictor.predict(
                text, 
                model_choice=model_choice,  # This was missing!
                include_probabilities=True
            )
            
            prediction = ml_result['prediction']
            confidence = ml_result['confidence']
            probabilities = ml_result['probabilities']
            model_used = ml_result['model_name']
            
            logger.info(f"✓ ML Prediction: {prediction} ({confidence:.3f}) using {model_used}")
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return Response({
                'error': f'Analysis failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        processing_time = time.time() - start_time
        
        # Detect crisis situation
        is_crisis = detect_crisis(prediction, confidence, text)
        
        # Create analysis record
        analysis = TextAnalysis.objects.create(
            text_input=text,
            text_length=len(text),
            prediction=prediction,
            confidence_score=confidence,
            probabilities=probabilities,
            model_used=model_choice,
            processing_time=processing_time,
            session_id=request.session.session_key or str(uuid.uuid4())[:8],
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')
        )
        
        # Generate response
        response_data = {
            'id': str(analysis.id),
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': model_used,
            'processing_time': round(processing_time, 3),
            'text_length': len(text),
            'message': generate_message(prediction, confidence),
            'recommendations': generate_student_recommendations(prediction),
            'is_crisis': is_crisis,
            'study_tips': generate_study_tips(prediction),
            'quick_actions': generate_quick_actions(prediction),
        }
        
        # Add crisis resources if needed
        if is_crisis:
            response_data['crisis_resources'] = get_crisis_resources()
        
        # Add explanation if requested
        if include_explanation:
            response_data['explanation'] = generate_explanation(prediction, confidence, text)
        
        logger.info(f"Analysis completed: {prediction} ({confidence:.3f}) - {len(text)} chars")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return Response({
            'error': 'Analysis failed. Please try again.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def submit_feedback(request):
    """Submit user feedback for an analysis"""
    try:
        analysis_id = request.data.get('analysis_id')
        rating = request.data.get('rating')
        feedback_text = request.data.get('feedback_text', '')
        is_helpful = request.data.get('is_helpful')
        
        if not analysis_id:
            return Response({'error': 'Analysis ID required'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not rating or rating not in [1, 2, 3, 4, 5]:
            return Response({'error': 'Rating must be 1-5'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            analysis = TextAnalysis.objects.get(id=analysis_id)
        except TextAnalysis.DoesNotExist:
            return Response({'error': 'Analysis not found'}, status=status.HTTP_404_NOT_FOUND)
        
        if hasattr(analysis, 'feedback'):
            return Response({'error': 'Feedback already submitted'}, status=status.HTTP_400_BAD_REQUEST)
        
        feedback = UserFeedback.objects.create(
            analysis=analysis,
            rating=rating,
            feedback_text=feedback_text,
            is_helpful=is_helpful
        )
        
        logger.info(f"Feedback submitted: {rating}/5 for analysis {analysis_id}")
        
        return Response({
            'success': True,
            'message': 'Thank you for your feedback!',
            'feedback_id': feedback.id
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return Response({
            'error': 'Failed to submit feedback'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def analytics_dashboard_data(request):
    """Get analytics data for dashboard"""
    try:
        total_analyses = TextAnalysis.objects.count()
        
        prediction_dist = TextAnalysis.objects.values('prediction').annotate(
            count=Count('prediction')
        ).order_by('-count')
        
        model_stats = TextAnalysis.objects.values('model_used').annotate(
            count=Count('model_used'),
            avg_confidence=Avg('confidence_score'),
            avg_time=Avg('processing_time')
        )
        
        feedback_count = UserFeedback.objects.count()
        avg_rating = UserFeedback.objects.aggregate(Avg('rating'))['rating__avg'] or 0
        
        week_ago = timezone.now() - timezone.timedelta(days=7)
        daily_counts = []
        for i in range(7):
            date = (timezone.now() - timezone.timedelta(days=i)).date()
            count = TextAnalysis.objects.filter(created_at__date=date).count()
            daily_counts.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': count
            })
        daily_counts.reverse()
        
        data = {
            'total_analyses': total_analyses,
            'feedback_count': feedback_count,
            'average_rating': round(avg_rating, 2),
            'feedback_rate': round((feedback_count / max(total_analyses, 1)) * 100, 1),
            'prediction_distribution': list(prediction_dist),
            'model_statistics': list(model_stats),
            'daily_activity': daily_counts
        }
        
        return Response(data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return Response({
            'error': 'Failed to fetch analytics'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def model_info(request):
    """Get information about the loaded ML model"""
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        
        return Response(info, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return Response({
            'error': 'Failed to get model information',
            'status': 'error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """API health check endpoint"""
    try:
        predictor = get_predictor()
        model_loaded = predictor.is_loaded
        
        db_healthy = TextAnalysis.objects.count() >= 0
        
        health_data = {
            'status': 'healthy' if (model_loaded and db_healthy) else 'degraded',
            'timestamp': timezone.now().isoformat(),
            'components': {
                'ml_model': 'loaded' if model_loaded else 'not_loaded',
                'database': 'connected' if db_healthy else 'error'
            }
        }
        
        status_code = status.HTTP_200_OK if health_data['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return Response(health_data, status=status_code)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# Helper Functions

def detect_crisis(prediction, confidence, text):
    """Detect if user might be in crisis"""
    crisis_keywords = ['suicide', 'kill myself', 'end it all', 'want to die', 'no reason to live']
    text_lower = text.lower()
    
    has_crisis_keyword = any(keyword in text_lower for keyword in crisis_keywords)
    high_concern = (prediction in ['depression', 'anxiety'] and confidence > 0.85)
    
    return has_crisis_keyword or high_concern


def get_crisis_resources():
    """Get crisis intervention resources"""
    return {
        'message': '⚠️ We noticed you might be going through a difficult time. Please reach out for help.',
        'hotlines': [
            {'name': 'National Suicide Prevention Lifeline (US)', 'number': '988', 'available': '24/7'},
            {'name': 'Crisis Text Line', 'number': 'Text HOME to 741741', 'available': '24/7'},
            {'name': 'International Association for Suicide Prevention', 'url': 'https://www.iasp.info/resources/Crisis_Centres/'},
        ],
        'immediate_actions': [
            'Talk to a trusted friend, family member, or counselor',
            'Call your campus mental health services',
            'Visit your nearest emergency room if you\'re in immediate danger',
            'Remove any items that could be used for self-harm'
        ]
    }


def generate_message(prediction, confidence):
    """Generate contextual message based on prediction"""
    
    messages = {
        'neutral': {
            'high': "Great! Your text suggests a balanced mental state. Keep up the good work with your studies and self-care! 📚✨",
            'medium': "Your text appears mostly positive. Continue maintaining healthy study-life balance.",
            'low': "The analysis suggests a generally stable state. Keep monitoring how you feel."
        },
        'depression': {
            'high': "We detected patterns that may indicate you're struggling. Please consider talking to a campus counselor or trusted advisor. Remember, seeking help is a sign of strength. 💙",
            'medium': "Some concerning patterns detected. It might help to talk to someone you trust about how you're feeling.",
            'low': "Your text contains some stress indicators. Consider reaching out to friends or campus resources."
        },
        'anxiety': {
            'high': "Your text suggests elevated stress levels - common during exam periods! Try our quick relaxation tools below and consider campus counseling services. 🧘",
            'medium': "Some stress indicators detected. Remember to take study breaks and practice self-care.",
            'low': "Mild stress patterns observed. This is normal for students - don't forget to breathe!"
        }
    }
    
    level = 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low'
    return messages.get(prediction, {}).get(level, "Analysis complete.")


def generate_student_recommendations(prediction):
    """Generate student-specific recommendations"""
    
    recommendations = {
        'neutral': [
            "Maintain your current study schedule - it's working!",
            "Join study groups to stay socially connected",
            "Keep a regular sleep schedule (7-9 hours)",
            "Schedule regular breaks during study sessions (Pomodoro technique)"
        ],
        'depression': [
            "🏫 Visit your campus counseling center (usually free for students)",
            "📞 Talk to your academic advisor about workload adjustments",
            "👥 Join student support groups or mental health clubs on campus",
            "💪 Maintain physical activity - even a 10-minute walk helps",
            "📝 Consider therapy apps like BetterHelp (student discounts available)"
        ],
        'anxiety': [
            "🧘 Try the 5-minute breathing exercise below before exams",
            "📅 Break large assignments into smaller, manageable tasks",
            "🏋️ Use campus gym facilities - exercise reduces anxiety",
            "😴 Avoid all-nighters - they increase stress hormones",
            "🎯 Practice exam anxiety techniques with campus resources"
        ]
    }
    
    return recommendations.get(prediction, [])


def generate_study_tips(prediction):
    """Study-specific tips based on mental state"""
    
    tips = {
        'neutral': [
            "Active Recall: Test yourself instead of re-reading notes",
            "Spaced Repetition: Review material at increasing intervals",
            "Teach Someone: Explaining concepts solidifies understanding"
        ],
        'depression': [
            "Start Small: Even 15 minutes of studying is progress",
            "Study With Others: Reduces isolation and increases motivation",
            "Reward System: Small rewards after completing tasks"
        ],
        'anxiety': [
            "Pre-Study Ritual: 5 deep breaths before starting",
            "Time-Boxing: Set a timer, study focused, then break",
            "Anxiety Journal: Write worries down before studying to clear mind"
        ]
    }
    
    return tips.get(prediction, [])


def generate_quick_actions(prediction):
    """Quick actionable items students can do right now"""
    
    actions = {
        'neutral': [
            {'icon': '📚', 'action': 'Plan tomorrow\'s study schedule', 'time': '5 min'},
            {'icon': '🎯', 'action': 'Set 3 achievable goals for this week', 'time': '3 min'},
            {'icon': '💧', 'action': 'Drink water and take a 5-min stretch break', 'time': '5 min'}
        ],
        'depression': [
            {'icon': '☀️', 'action': 'Go outside for 10 minutes (natural light helps)', 'time': '10 min'},
            {'icon': '📞', 'action': 'Text a friend or family member', 'time': '2 min'},
            {'icon': '🎵', 'action': 'Listen to uplifting music', 'time': '5 min'}
        ],
        'anxiety': [
            {'icon': '🧘', 'action': 'Try box breathing exercise (link below)', 'time': '3 min'},
            {'icon': '📝', 'action': 'Write down your top worry and one step to address it', 'time': '5 min'},
            {'icon': '🏃', 'action': 'Do 20 jumping jacks (releases tension)', 'time': '2 min'}
        ]
    }
    
    return actions.get(prediction, [])


def generate_explanation(prediction, confidence, text):
    """Generate explanation for the prediction"""
    
    explanations = {
        'depression': "The AI detected language patterns common in students experiencing low mood, such as expressions of hopelessness, academic overwhelm, or loss of interest in studies.",
        'anxiety': "The analysis identified linguistic markers typically associated with academic stress, including expressions of worry about grades, deadlines, or performance pressure.",
        'neutral': "Your language patterns suggest you're managing stress well and maintaining a balanced perspective on your academic responsibilities."
    }
    
    confidence_explanation = f"The model's confidence in this prediction is {confidence:.1%}. "
    
    if confidence > 0.8:
        confidence_explanation += "This indicates a strong pattern match with our training data."
    elif confidence > 0.6:
        confidence_explanation += "This suggests a moderate pattern match with some uncertainty."
    else:
        confidence_explanation += "This indicates lower certainty, suggesting mixed patterns in your text."
    
    return {
        'reasoning': explanations.get(prediction, "Analysis complete."),
        'confidence_explanation': confidence_explanation,
        'model_details': f"Using AI trained on student mental health data with 85%+ accuracy.",
        'disclaimer': '⚠️ This is an AI tool for awareness, not a medical diagnosis. Always consult campus health services for professional support.'
    }


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
