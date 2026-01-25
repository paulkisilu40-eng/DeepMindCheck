"""
Wellness Plan Generator Logic
Generates personalized 7-day plans based on mental state
"""
import random
from datetime import datetime, timedelta

def generate_wellness_plan(mental_state, confidence):
    """
    Generate a 7-day wellness plan based on detected mental state
    """
    plan = {
        'mental_state': mental_state,
        'confidence': confidence,
        'created_at': datetime.now().strftime('%Y-%m-%d'),
        'days': []
    }
    
    # Define activities based on state
    activities = get_activities_for_state(mental_state)
    
    for i in range(7):
        day_date = datetime.now() + timedelta(days=i)
        day_plan = {
            'day': i + 1,
            'date': day_date.strftime('%A, %b %d'),
            'focus': get_daily_focus(mental_state, i),
            'tasks': [
                get_morning_routine(mental_state),
                activities[i % len(activities)],
                get_study_task(mental_state),
                get_evening_routine(mental_state)
            ]
        }
        plan['days'].append(day_plan)
        
    return plan

def get_activities_for_state(state):
    """Return list of state-specific activities"""
    if state == 'depression':
        return [
            {'time': '15 min', 'icon': '🚶', 'text': 'Take a short walk outside (nature helps)'},
            {'time': '10 min', 'icon': '🎵', 'text': 'Listen to your favorite uplifting playlist'},
            {'time': '20 min', 'icon': '🎨', 'text': 'Do something creative (draw, write, color)'},
            {'time': '15 min', 'icon': '🧹', 'text': 'Tidy up one small area of your room'},
            {'time': '30 min', 'icon': '👥', 'text': 'Call or text a friend/family member'},
            {'time': '20 min', 'icon': '🛁', 'text': 'Take a relaxing warm shower or bath'},
            {'time': '15 min', 'icon': '📓', 'text': 'Write down 3 things you are grateful for'}
        ]
    elif state == 'anxiety':
        return [
            {'time': '10 min', 'icon': '🧘', 'text': 'Practice box breathing (4-4-4-4)'},
            {'time': '15 min', 'icon': '📝', 'text': 'Write down your worries ("Brain Dump")'},
            {'time': '20 min', 'icon': '🏃', 'text': 'Do some physical exercise to release tension'},
            {'time': '10 min', 'icon': '📵', 'text': 'Take a digital detox break'},
            {'time': '15 min', 'icon': '🍵', 'text': 'Drink herbal tea and sit quietly'},
            {'time': '20 min', 'icon': '🧩', 'text': 'Focus on a puzzle or engaging game'},
            {'time': '10 min', 'icon': '🌿', 'text': 'Practice 5-4-3-2-1 grounding technique'}
        ]
    else: # neutral/wellness
        return [
            {'time': '20 min', 'icon': '📚', 'text': 'Read a book for pleasure (not study)'},
            {'time': '30 min', 'icon': '🎯', 'text': 'Set goals for the upcoming week'},
            {'time': '20 min', 'icon': '🍳', 'text': 'Cook a healthy meal'},
            {'time': '30 min', 'icon': '🤸', 'text': 'Try a new workout or activity'},
            {'time': '15 min', 'icon': '🧘', 'text': 'Practice mindfulness meditation'},
            {'time': '30 min', 'icon': '🎬', 'text': 'Watch an episode of your favorite show'},
            {'time': '20 min', 'icon': '🧹', 'text': 'Organize your study space'}
        ]

def get_morning_routine(state):
    if state == 'depression':
        return {'time': 'Morning', 'icon': '☀️', 'text': 'Open curtains immediately for light'}
    elif state == 'anxiety':
        return {'time': 'Morning', 'icon': '💧', 'text': 'Drink water and stretch for 5 mins'}
    return {'time': 'Morning', 'icon': '🌅', 'text': 'Plan your top 3 priorities today'}

def get_evening_routine(state):
    return {'time': 'Evening', 'icon': '🌙', 'text': 'No screens 30 mins before bed'}

def get_study_task(state):
    if state == 'depression':
        return {'time': 'Study', 'icon': '⏱️', 'text': 'Study for just 20 mins (Pomodoro)'}
    elif state == 'anxiety':
        return {'time': 'Study', 'icon': '📝', 'text': 'Break big tasks into tiny steps'}
    return {'time': 'Study', 'icon': '🧠', 'text': 'Review notes using active recall'}

def get_daily_focus(state, day_index):
    focuses = {
        'depression': ['Self-Compassion', 'Movement', 'Connection', 'Hygiene', 'Nature', 'Creativity', 'Reflection'],
        'anxiety': ['Grounding', 'Breathe', 'Focus', 'Release', 'Calm', 'Present', 'Rest'],
        'neutral': ['Productivity', 'Growth', 'Balance', 'Social', 'Health', 'Learning', 'Review']
    }
    return focuses.get(state, focuses['neutral'])[day_index % 7]
