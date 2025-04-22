from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Event
from .forms import EventForm
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
from django.http import JsonResponse
from django.utils import timezone
import json
from datetime import datetime
import pandas as pd
from django.contrib import messages
from .simple_model import train_model, predict
import traceback

# Global variable to store the trained models
GLOBAL_MODEL = None

def initialize_model():
    """Initialize the models at startup"""
    global GLOBAL_MODEL
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'historical_costs.csv')
        
        # Train models
        model_data = train_model(csv_path)
        if model_data is None:
            raise Exception("Failed to train models")
        
        # Store the models globally
        GLOBAL_MODEL = model_data
        print("Models trained successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        traceback.print_exc()
        return False

# Initialize the models when the module is loaded
initialize_model()

class PredictionsView(LoginRequiredMixin, TemplateView):
    template_name = 'predictor/predictions.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'historical_costs.csv')
        
        # Load historical data
        try:
            df = pd.read_csv(csv_path)
            df = df[['Year', 'Average Monthly Rent (England) (£)', 'Petrol Price (£/litre)', 'Weekly Food Expenditure (£)']]
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            df = df.sort_values('Year')
            
            # Calculate percentage changes for rent
            df['Rent_Change'] = df['Average Monthly Rent (England) (£)'].pct_change() * 100
            
            historical_data = []
            for _, row in df.iterrows():
                historical_data.append({
                    'year': int(row['Year']),
                    'rent': round(row['Average Monthly Rent (England) (£)'], 2),
                    'petrol': round(row['Petrol Price (£/litre)'], 2),
                    'food': round(row['Weekly Food Expenditure (£)'], 2),
                    'rent_change': round(row['Rent_Change'], 1) if not pd.isna(row['Rent_Change']) else None
                })
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            traceback.print_exc()
            historical_data = []

        # Generate future years for prediction
        latest_year = max(item['year'] for item in historical_data) if historical_data else 2023
        future_years = list(range(latest_year + 1, latest_year + 6))
        
        context.update({
            'historical_data': historical_data,
            'future_years': future_years,
            'prediction_results': None,
            'selected_year': None
        })
        return context
    
    def post(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        
        try:
            selected_year = int(request.POST.get('year'))
            context['selected_year'] = selected_year
            
            # Use the trained models
            if GLOBAL_MODEL is None:
                if not initialize_model():
                    raise Exception("Failed to initialize models")
            
            # Make prediction using linear regression models
            predictions = predict(selected_year, GLOBAL_MODEL)
            if predictions is None:
                raise Exception("Failed to generate predictions")
            
            context['prediction_results'] = {
                'rent': round(predictions['rent'], 2),
                'petrol': round(predictions['petrol'], 2),
                'food': round(predictions['food'], 2)
            }
            print(f"Generated predictions for year {selected_year}: {context['prediction_results']}")
            
        except ValueError as e:
            print(f"Invalid year selected: {str(e)}")
            messages.error(request, "Please select a valid year.")
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            traceback.print_exc()
            messages.error(request, "An error occurred while generating predictions.")
        
        return self.render_to_response(context)

@login_required
def home(request):
    # Get upcoming events for the next 7 days
    now = timezone.now()
    upcoming_events = Event.objects.filter(
        date__gte=now.date(),
        date__lte=now.date() + timezone.timedelta(days=7)
    ).order_by('date')
    
    return render(request, 'predictor/home.html', {'upcoming_events': upcoming_events})

@login_required
def event_list(request):
    events = Event.objects.all().order_by('date')
    return render(request, 'predictor/event_list.html', {'events': events})

@login_required
def event_create(request):
    if request.method == 'POST':
        form = EventForm(request.POST)
        if form.is_valid():
            event = form.save()
            messages.success(request, 'Event created successfully.')
            return redirect('event_list')
    else:
        form = EventForm()
    return render(request, 'predictor/event_form.html', {'form': form, 'title': 'Create Event'})

@login_required
def event_update(request, event_id):
    event = get_object_or_404(Event, id=event_id)
    if request.method == 'POST':
        form = EventForm(request.POST, instance=event)
        if form.is_valid():
            form.save()
            messages.success(request, 'Event updated successfully.')
            return redirect('event_list')
    else:
        form = EventForm(instance=event)
    return render(request, 'predictor/event_form.html', {'form': form, 'title': 'Update Event'})

@login_required
def profile(request):
    return render(request, 'predictor/profile.html')

@login_required
def dashboard(request):
    # Load historical data for charts
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'historical_costs.csv')
    
    try:
        df = pd.read_csv(csv_path)
        df = df[['Year', 'Average Monthly Rent (England) (£)', 'Petrol Price (£/litre)', 'Weekly Food Expenditure (£)']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df = df.sort_values('Year')
        
        # Prepare historical data
        historical_data = {
            'rent': {},
            'petrol': {},
            'food': {}
        }
        
        for _, row in df.iterrows():
            year = str(int(row['Year']))
            historical_data['rent'][year] = round(row['Average Monthly Rent (England) (£)'], 2)
            historical_data['petrol'][year] = round(row['Petrol Price (£/litre)'], 2)
            historical_data['food'][year] = round(row['Weekly Food Expenditure (£)'], 2)
        
        # Get predictions for next 5 years
        latest_year = df['Year'].max()
        predicted_data = {
            'rent': {},
            'petrol': {},
            'food': {}
        }
        
        if GLOBAL_MODEL is None:
            initialize_model()
        
        for year in range(int(latest_year) + 1, int(latest_year) + 6):
            predictions = predict(year, GLOBAL_MODEL)
            if predictions:
                predicted_data['rent'][str(year)] = round(predictions['rent'], 2)
                predicted_data['petrol'][str(year)] = round(predictions['petrol'], 2)
                predicted_data['food'][str(year)] = round(predictions['food'], 2)
        
        # Prepare data for the template
        chart_data = {
            'historical': historical_data,
            'predicted': predicted_data
        }
        
        # Get recent predictions for the table
        recent_predictions = []
        latest_values = {
            'rent': df['Average Monthly Rent (England) (£)'].iloc[-1],
            'petrol': df['Petrol Price (£/litre)'].iloc[-1],
            'food': df['Weekly Food Expenditure (£)'].iloc[-1]
        }
        
        # Add latest historical values
        recent_predictions.extend([
            {
                'year': int(latest_year),
                'cost': latest_values['rent'],
                'change': 0,
                'is_prediction': False
            },
            {
                'year': int(latest_year),
                'cost': latest_values['petrol'],
                'change': 0,
                'is_prediction': False
            },
            {
                'year': int(latest_year),
                'cost': latest_values['food'],
                'change': 0,
                'is_prediction': False
            }
        ])
        
        # Add predictions
        next_year = int(latest_year) + 1
        if GLOBAL_MODEL:
            predictions = predict(next_year, GLOBAL_MODEL)
            if predictions:
                recent_predictions.extend([
                    {
                        'year': next_year,
                        'cost': predictions['rent'],
                        'change': ((predictions['rent'] - latest_values['rent']) / latest_values['rent']) * 100,
                        'is_prediction': True
                    },
                    {
                        'year': next_year,
                        'cost': predictions['petrol'],
                        'change': ((predictions['petrol'] - latest_values['petrol']) / latest_values['petrol']) * 100,
                        'is_prediction': True
                    },
                    {
                        'year': next_year,
                        'cost': predictions['food'],
                        'change': ((predictions['food'] - latest_values['food']) / latest_values['food']) * 100,
                        'is_prediction': True
                    }
                ])
        
        context = {
            'chart_data': json.dumps(chart_data),
            'recent_predictions': recent_predictions
        }
        
        return render(request, 'predictor/dashboard.html', context)
        
    except Exception as e:
        print(f"Error preparing dashboard data: {str(e)}")
        traceback.print_exc()
        messages.error(request, "An error occurred while preparing the dashboard.")
        return render(request, 'predictor/dashboard.html', {'error': True})

