"""
Wellness Plan Generator for DeepMindCheck
Generates personalized 7-day wellness plans based on mental state
"""

from datetime import datetime, timedelta


def generate_wellness_plan(mental_state, confidence=0.8):
    """
    Generate a personalized 7-day wellness plan based on mental state
    
    Args:
        mental_state (str): The detected mental state ('neutral', 'depression', 'anxiety')
        confidence (float): Confidence score from the model (0-1)
    
    Returns:
        dict: A structured wellness plan with days and tasks
    """
    
    # Calculate dates for the 7-day plan
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%B %d, %Y') for i in range(7)]
    
    # Define plans based on mental state
    plans = {
        'depression': {
            'days': [
                {
                    'day': 1,
                    'focus': 'Gentle Start',
                    'date': dates[0],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '☀️', 'text': 'Open curtains and get 5 minutes of natural light'},
                        {'time': '10:00 AM', 'icon': '🚿', 'text': 'Take a refreshing shower'},
                        {'time': '12:00 PM', 'icon': '🥗', 'text': 'Eat a nutritious meal (even if small)'},
                        {'time': '3:00 PM', 'icon': '🚶', 'text': '5-minute walk outside (just around the block)'},
                        {'time': '8:00 PM', 'icon': '📱', 'text': 'Text one friend or family member'},
                    ]
                },
                {
                    'day': 2,
                    'focus': 'Building Momentum',
                    'date': dates[1],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '💧', 'text': 'Drink a full glass of water'},
                        {'time': '10:00 AM', 'icon': '🧘', 'text': '5-minute gentle stretching'},
                        {'time': '2:00 PM', 'icon': '📚', 'text': 'Read for 10 minutes (anything enjoyable)'},
                        {'time': '4:00 PM', 'icon': '🎵', 'text': 'Listen to uplifting music for 15 minutes'},
                        {'time': '9:00 PM', 'icon': '📝', 'text': 'Write down one positive thing from today'},
                    ]
                },
                {
                    'day': 3,
                    'focus': 'Social Connection',
                    'date': dates[2],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '☕', 'text': 'Have breakfast (even if small)'},
                        {'time': '11:00 AM', 'icon': '👥', 'text': 'Reach out to a friend for a short chat'},
                        {'time': '2:00 PM', 'icon': '🚶', 'text': '10-minute walk in nature or park'},
                        {'time': '5:00 PM', 'icon': '🎨', 'text': 'Try a creative activity (draw, color, write)'},
                        {'time': '8:00 PM', 'icon': '🛀', 'text': 'Take a relaxing bath or shower'},
                    ]
                },
                {
                    'day': 4,
                    'focus': 'Physical Care',
                    'date': dates[3],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🥤', 'text': 'Make a healthy smoothie or breakfast'},
                        {'time': '10:00 AM', 'icon': '🏃', 'text': '15-minute light exercise (walk, yoga, dance)'},
                        {'time': '1:00 PM', 'icon': '🌿', 'text': 'Spend time in nature or with plants'},
                        {'time': '4:00 PM', 'icon': '📖', 'text': 'Read something motivational or watch TED talk'},
                        {'time': '9:00 PM', 'icon': '😴', 'text': 'Prepare for bed early (good sleep hygiene)'},
                    ]
                },
                {
                    'day': 5,
                    'focus': 'Routine Building',
                    'date': dates[4],
                    'tasks': [
                        {'time': '7:30 AM', 'icon': '⏰', 'text': 'Wake up at consistent time'},
                        {'time': '12:00 PM', 'icon': '🍽️', 'text': 'Eat lunch at a regular time'},
                        {'time': '3:00 PM', 'icon': '🎯', 'text': 'Complete one small productive task'},
                        {'time': '6:00 PM', 'icon': '👨‍👩‍👧', 'text': 'Connect with family or attend social event'},
                        {'time': '10:00 PM', 'icon': '📱', 'text': 'Put devices away for better sleep'},
                    ]
                },
                {
                    'day': 6,
                    'focus': 'Self-Compassion',
                    'date': dates[5],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '🙏', 'text': 'Practice 5 minutes of gratitude or meditation'},
                        {'time': '11:00 AM', 'icon': '💆', 'text': 'Do something nurturing for yourself'},
                        {'time': '2:00 PM', 'icon': '🎨', 'text': 'Engage in a hobby you enjoy'},
                        {'time': '5:00 PM', 'icon': '📞', 'text': 'Have a meaningful conversation with someone'},
                        {'time': '8:00 PM', 'icon': '📝', 'text': 'Reflect on your progress this week'},
                    ]
                },
                {
                    'day': 7,
                    'focus': 'Looking Forward',
                    'date': dates[6],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '🌅', 'text': 'Plan something to look forward to next week'},
                        {'time': '11:00 AM', 'icon': '🏃', 'text': '20-minute activity you enjoy'},
                        {'time': '2:00 PM', 'icon': '💪', 'text': 'Acknowledge your progress and strength'},
                        {'time': '5:00 PM', 'icon': '🎉', 'text': 'Celebrate completing the week (small treat)'},
                        {'time': '9:00 PM', 'icon': '📋', 'text': 'Consider continuing or seeking professional support'},
                    ]
                },
            ]
        },
        'anxiety': {
            'days': [
                {
                    'day': 1,
                    'focus': 'Grounding & Calm',
                    'date': dates[0],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🧘', 'text': '5-4-3-2-1 grounding exercise'},
                        {'time': '10:00 AM', 'icon': '🫁', 'text': 'Box breathing: 4 counts in, hold, out, hold'},
                        {'time': '1:00 PM', 'icon': '🥗', 'text': 'Eat mindfully without distractions'},
                        {'time': '4:00 PM', 'icon': '🚶', 'text': 'Gentle walk focusing on your senses'},
                        {'time': '8:00 PM', 'icon': '📝', 'text': 'Write down your worries, then set them aside'},
                    ]
                },
                {
                    'day': 2,
                    'focus': 'Stress Management',
                    'date': dates[1],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '☕', 'text': 'Limit caffeine intake today'},
                        {'time': '10:00 AM', 'icon': '💆', 'text': 'Progressive muscle relaxation (10 min)'},
                        {'time': '2:00 PM', 'icon': '🎵', 'text': 'Listen to calming music or nature sounds'},
                        {'time': '5:00 PM', 'icon': '✅', 'text': 'Break down one worry into small action steps'},
                        {'time': '9:00 PM', 'icon': '📱', 'text': 'Screen-free wind down routine'},
                    ]
                },
                {
                    'day': 3,
                    'focus': 'Mind-Body Connection',
                    'date': dates[2],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🌅', 'text': '5-minute morning meditation'},
                        {'time': '11:00 AM', 'icon': '🧘', 'text': 'Gentle yoga or stretching (15 min)'},
                        {'time': '2:00 PM', 'icon': '🫁', 'text': 'Practice diaphragmatic breathing'},
                        {'time': '5:00 PM', 'icon': '🌿', 'text': 'Spend time in nature or with plants'},
                        {'time': '8:00 PM', 'icon': '🛀', 'text': 'Relaxing bath with calming scents'},
                    ]
                },
                {
                    'day': 4,
                    'focus': 'Positive Distraction',
                    'date': dates[3],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '🎨', 'text': 'Engage in a creative hobby'},
                        {'time': '12:00 PM', 'icon': '👥', 'text': 'Connect with a supportive friend'},
                        {'time': '3:00 PM', 'icon': '📚', 'text': 'Read something engaging or watch comedy'},
                        {'time': '6:00 PM', 'icon': '🏃', 'text': 'Physical activity to release tension'},
                        {'time': '9:00 PM', 'icon': '🙏', 'text': 'Gratitude journaling (3 things)'},
                    ]
                },
                {
                    'day': 5,
                    'focus': 'Healthy Boundaries',
                    'date': dates[4],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🚫', 'text': 'Say no to one non-essential commitment'},
                        {'time': '11:00 AM', 'icon': '⏰', 'text': 'Schedule breaks throughout your day'},
                        {'time': '2:00 PM', 'icon': '📱', 'text': 'Take a social media break'},
                        {'time': '5:00 PM', 'icon': '🎯', 'text': 'Focus on what you can control only'},
                        {'time': '8:00 PM', 'icon': '😴', 'text': 'Establish calming bedtime routine'},
                    ]
                },
                {
                    'day': 6,
                    'focus': 'Self-Compassion',
                    'date': dates[5],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '💭', 'text': 'Challenge one anxious thought with evidence'},
                        {'time': '12:00 PM', 'icon': '💪', 'text': 'Acknowledge your strength in managing anxiety'},
                        {'time': '3:00 PM', 'icon': '🎨', 'text': 'Do something just for enjoyment'},
                        {'time': '6:00 PM', 'icon': '👥', 'text': 'Share feelings with trusted person'},
                        {'time': '9:00 PM', 'icon': '📝', 'text': 'List coping strategies that worked this week'},
                    ]
                },
                {
                    'day': 7,
                    'focus': 'Moving Forward',
                    'date': dates[6],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '🌟', 'text': 'Celebrate managing anxiety this week'},
                        {'time': '11:00 AM', 'icon': '📋', 'text': 'Create anxiety action plan for next week'},
                        {'time': '2:00 PM', 'icon': '🧘', 'text': 'Practice favorite relaxation technique'},
                        {'time': '5:00 PM', 'icon': '🎯', 'text': 'Set one realistic goal for next week'},
                        {'time': '8:00 PM', 'icon': '💭', 'text': 'Consider if professional support would help'},
                    ]
                },
            ]
        },
        'neutral': {
            'days': [
                {
                    'day': 1,
                    'focus': 'Wellness Foundation',
                    'date': dates[0],
                    'tasks': [
                        {'time': '7:00 AM', 'icon': '🌅', 'text': 'Start day with 5-minute meditation'},
                        {'time': '9:00 AM', 'icon': '🥗', 'text': 'Eat a balanced breakfast'},
                        {'time': '12:00 PM', 'icon': '💧', 'text': 'Drink 8 glasses of water throughout day'},
                        {'time': '6:00 PM', 'icon': '🏃', 'text': '30-minute exercise or walk'},
                        {'time': '10:00 PM', 'icon': '😴', 'text': 'Get 7-8 hours of quality sleep'},
                    ]
                },
                {
                    'day': 2,
                    'focus': 'Mental Clarity',
                    'date': dates[1],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '📝', 'text': 'Journal for 10 minutes'},
                        {'time': '10:00 AM', 'icon': '🎯', 'text': 'Set 3 priorities for the day'},
                        {'time': '2:00 PM', 'icon': '🧠', 'text': 'Take a 15-minute mindfulness break'},
                        {'time': '5:00 PM', 'icon': '📚', 'text': 'Read or learn something new (20 min)'},
                        {'time': '9:00 PM', 'icon': '📱', 'text': 'Digital detox 1 hour before bed'},
                    ]
                },
                {
                    'day': 3,
                    'focus': 'Social Wellness',
                    'date': dates[2],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '👥', 'text': 'Reach out to a friend or family member'},
                        {'time': '12:00 PM', 'icon': '🤝', 'text': 'Have a meaningful conversation'},
                        {'time': '3:00 PM', 'icon': '😊', 'text': 'Perform one act of kindness'},
                        {'time': '6:00 PM', 'icon': '🎉', 'text': 'Plan a social activity for this week'},
                        {'time': '8:00 PM', 'icon': '🙏', 'text': 'Express gratitude to someone'},
                    ]
                },
                {
                    'day': 4,
                    'focus': 'Physical Vitality',
                    'date': dates[3],
                    'tasks': [
                        {'time': '7:00 AM', 'icon': '🧘', 'text': '15-minute yoga or stretching'},
                        {'time': '9:00 AM', 'icon': '🥤', 'text': 'Make a nutritious smoothie'},
                        {'time': '1:00 PM', 'icon': '🚶', 'text': 'Take a walk in nature (30 min)'},
                        {'time': '5:00 PM', 'icon': '🏋️', 'text': 'Strength training or active hobby'},
                        {'time': '9:00 PM', 'icon': '🛀', 'text': 'Relaxing self-care routine'},
                    ]
                },
                {
                    'day': 5,
                    'focus': 'Creative Expression',
                    'date': dates[4],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🎨', 'text': 'Try a creative activity (draw, write, music)'},
                        {'time': '11:00 AM', 'icon': '💭', 'text': 'Brainstorm new ideas or goals'},
                        {'time': '2:00 PM', 'icon': '🎵', 'text': 'Listen to inspiring music or podcast'},
                        {'time': '5:00 PM', 'icon': '📸', 'text': 'Capture beauty around you (photos/notes)'},
                        {'time': '8:00 PM', 'icon': '✍️', 'text': 'Write about your day creatively'},
                    ]
                },
                {
                    'day': 6,
                    'focus': 'Growth & Learning',
                    'date': dates[5],
                    'tasks': [
                        {'time': '8:00 AM', 'icon': '🎓', 'text': 'Learn something new (online course, skill)'},
                        {'time': '11:00 AM', 'icon': '📖', 'text': 'Read for personal development'},
                        {'time': '2:00 PM', 'icon': '🎯', 'text': 'Work toward a personal goal'},
                        {'time': '5:00 PM', 'icon': '🤔', 'text': 'Reflect on lessons learned this week'},
                        {'time': '9:00 PM', 'icon': '📝', 'text': 'Plan next week\'s growth activities'},
                    ]
                },
                {
                    'day': 7,
                    'focus': 'Rest & Recharge',
                    'date': dates[6],
                    'tasks': [
                        {'time': '9:00 AM', 'icon': '🌄', 'text': 'Sleep in or have leisurely morning'},
                        {'time': '11:00 AM', 'icon': '😌', 'text': 'Do something purely for enjoyment'},
                        {'time': '2:00 PM', 'icon': '🌿', 'text': 'Spend time in nature or outdoors'},
                        {'time': '5:00 PM', 'icon': '🎊', 'text': 'Celebrate your wellness achievements'},
                        {'time': '8:00 PM', 'icon': '🔮', 'text': 'Set intentions for the week ahead'},
                    ]
                },
            ]
        }
    }
    
    # Get the appropriate plan or default to neutral
    plan_data = plans.get(mental_state.lower(), plans['neutral'])
    
    # Add metadata
    plan = {
        'mental_state': mental_state.title(),
        'confidence': confidence,
        'generated_date': today.strftime('%B %d, %Y'),
        'days': plan_data['days']
    }
    
    return plan
