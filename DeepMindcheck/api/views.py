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
            
            ml_result = predictor.predict(
                text, 
                model_choice=model_choice,
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
        
        # Create analysis record with user if authenticated
        analysis = TextAnalysis.objects.create(
            user=request.user if request.user.is_authenticated else None,  # ← FIXED!
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
        
        logger.info(f"Analysis saved - ID: {analysis.id}, User: {request.user if request.user.is_authenticated else 'Anonymous'}")
        
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
