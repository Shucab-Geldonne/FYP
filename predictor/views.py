from django.shortcuts import render, redirect
from django.views.generic import ListView, TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import CostOfLivingData, Event
from .forms import CostOfLivingForm
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
from django.http import JsonResponse
from django.utils import timezone
import json
from datetime import datetime
import pandas as pd
from django.contrib import messages
from sklearn.preprocessing import MinMaxScaler
from numpy.core.multiarray import _reconstruct
from numpy.dtypes import Float32DType, Float64DType
import joblib
from .ml_utils import CostPredictor
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

def load_model():
    try:
        # Get the absolute path to the model file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'predictor', 'models')
        model_path = os.path.join(model_dir, 'cost_predictor.pth')
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            print("Training new model...")
            if train_and_save_model():
                print("Model trained successfully")
            else:
                print("Failed to train model")
                return None
        
        # Load the model
        predictor = CostPredictor()
        predictor.load_model(model_path)
        return predictor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None

def predict_cost(model, year, scaler_X, scalers_y):
    try:
        # Normalize year using saved scaler
        normalized_year = scaler_X.transform([[year]])[0][0]
        
        # Convert to tensor
        x = torch.FloatTensor([[normalized_year]])
        
        # Make prediction
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            predictions = model(x)
            
            # Denormalize predictions using saved scalers
            rent_pred = scalers_y['rent'].inverse_transform(predictions[:, 0].numpy().reshape(-1, 1))[0][0]
            petrol_pred = scalers_y['petrol'].inverse_transform(predictions[:, 1].numpy().reshape(-1, 1))[0][0]
            food_pred = scalers_y['food'].inverse_transform(predictions[:, 2].numpy().reshape(-1, 1))[0][0]
            
            return {
                'rent': round(float(rent_pred), 2),
                'petrol': round(float(petrol_pred), 2),
                'food': round(float(food_pred), 2)
            }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        traceback.print_exc()
        return None

def train_and_save_model():
    try:
        # Get absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'historical_costs.csv')
        model_dir = os.path.join(base_dir, 'predictor', 'models')
        model_path = os.path.join(model_dir, 'cost_predictor.pth')
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize and train the model
        predictor = CostPredictor()
        predictor.train_model(csv_path)
        predictor.save_model(model_path)
        
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        traceback.print_exc()
        return False

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
            traceback.print_exc()
            messages.error(request, "Invalid year selected")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            messages.error(request, "An error occurred while making the prediction. Please try again.")
        
        return self.render_to_response(context)

@login_required
def predictions(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'historical_costs.csv')
    model_path = os.path.join(base_dir, 'predictor', 'models', 'cost_predictor.pth')
    
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
                'rent': row['Average Monthly Rent (England) (£)'],
                'petrol': row['Petrol Price (£/litre)'],
                'food': row['Weekly Food Expenditure (£)'],
                'rent_change': round(row['Rent_Change'], 1) if not pd.isna(row['Rent_Change']) else None
            })
    except Exception as e:
        messages.error(request, f"Error loading historical data: {str(e)}")
        historical_data = []

    # Generate future years for prediction
    latest_year = max(item['year'] for item in historical_data) if historical_data else 2023
    future_years = list(range(latest_year + 1, latest_year + 6))
    
    context = {
        'historical_data': historical_data,
        'future_years': future_years,
        'prediction_results': None,
        'selected_year': None
    }

    if request.method == 'POST':
        selected_year = int(request.POST.get('year'))
        context['selected_year'] = selected_year
        
        try:
            # Load or train model
            if not os.path.exists(model_path):
                print("Training new model...")
                train_and_save_model()
            
            # Load model and make prediction
            predictor = load_model()
            if predictor:
                predictions = predictor.predict(selected_year)
                context['prediction_results'] = {
                    'rent': round(predictions[0], 2),
                    'petrol': round(predictions[1], 2),
                    'food': round(predictions[2], 2)
                }
            else:
                messages.error(request, "Error loading the prediction model.")
        except Exception as e:
            messages.error(request, f"Error making prediction: {str(e)}")
            print(f"Error details: {str(e)}")

    return render(request, 'predictor/predictions.html', context)

@login_required
def predict_for_year(request, year):
    try:
        # Load the model and scalers
        model, scaler_X, scalers_y = load_model()
        if model is None:
            return JsonResponse({'error': 'Failed to load model'}, status=500)
        
        # Make prediction
        predictions = predict_cost(model, int(year), scaler_X, scalers_y)
        if predictions is None:
            return JsonResponse({'error': 'Failed to make prediction'}, status=500)
        
        return JsonResponse({
            'year': year,
            'predictions': predictions
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/accounts/login/')
def home(request):
    # Get upcoming events for the next 7 days
    upcoming_events = Event.objects.filter(
        start_date__gte=timezone.now(),
        start_date__lte=timezone.now() + timezone.timedelta(days=7)
    ).order_by('start_date')
    
    return render(request, 'predictor/home.html', {
        'upcoming_events': upcoming_events
    })

@login_required
def event_list(request):
    if request.method == 'GET':
        start = request.GET.get('start')
        end = request.GET.get('end')
        
        events = Event.objects.filter(user=request.user)
        if start and end:
            events = events.filter(start_date__gte=start, end_date__lte=end)
        
        event_list = []
        for event in events:
            event_list.append({
                'id': event.id,
                'title': event.title,
                'start': event.start_date.isoformat(),
                'end': event.end_date.isoformat(),
                'description': event.description,
                'color': event.color,
            })
        return JsonResponse(event_list, safe=False)

@login_required
def event_create(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        event = Event.objects.create(
            user=request.user,
            title=data.get('title'),
            description=data.get('description', ''),
            start_date=datetime.fromisoformat(data.get('start').replace('Z', '+00:00')),
            end_date=datetime.fromisoformat(data.get('end').replace('Z', '+00:00')),
            color=data.get('color', '#3788d8')
        )
        return JsonResponse({
            'id': event.id,
            'title': event.title,
            'start': event.start_date.isoformat(),
            'end': event.end_date.isoformat(),
            'description': event.description,
            'color': event.color,
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def event_update(request, event_id):
    try:
        event = Event.objects.get(id=event_id, user=request.user)
        if request.method == 'POST':
            data = json.loads(request.body)
            event.title = data.get('title', event.title)
            event.description = data.get('description', event.description)
            event.start_date = datetime.fromisoformat(data.get('start', event.start_date.isoformat()).replace('Z', '+00:00'))
            event.end_date = datetime.fromisoformat(data.get('end', event.end_date.isoformat()).replace('Z', '+00:00'))
            event.color = data.get('color', event.color)
            event.save()
            return JsonResponse({'status': 'success'})
        elif request.method == 'DELETE':
            event.delete()
            return JsonResponse({'status': 'success'})
    except Event.DoesNotExist:
        return JsonResponse({'error': 'Event not found'}, status=404)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def profile(request):
    return render(request, 'predictor/profile.html')

@login_required
def dashboard(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'historical_costs.csv')
    model_path = os.path.join(base_dir, 'predictor', 'models', 'simple_cost_predictor.pth')
    
    try:
        # Load historical data
        df = pd.read_csv(csv_path)
        df = df[['Year', 'Average Monthly Rent (England) (£)', 'Petrol Price (£/litre)', 'Weekly Food Expenditure (£)']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df = df.sort_values('Year')
        
        # Prepare historical data
        historical_data = {}
        for _, row in df.iterrows():
            year = str(int(row['Year']))
            historical_data[year] = float(row['Average Monthly Rent (England) (£)'])
        
        # Get predictions for future years
        predicted_data = {}
        latest_year = max(int(year) for year in historical_data.keys())
        future_years = range(latest_year + 1, latest_year + 6)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            model = SimpleNN()
            model.load_state_dict(checkpoint['model_state_dict'])
            X_scaler = checkpoint['X_scaler']
            scalers = (checkpoint['rent_scaler'], checkpoint['petrol_scaler'], checkpoint['food_scaler'])
            
            for year in future_years:
                predictions = predict(model, year, X_scaler, scalers)
                predicted_data[str(year)] = float(predictions['rent'])
        
        # Prepare data for the chart
        chart_data = {
            'historical': historical_data,
            'predicted': predicted_data
        }
        
        # Prepare recent predictions
        recent_predictions = []
        
        # Add historical data
        for year, rent in historical_data.items():
            year_int = int(year)
            if year_int >= latest_year - 2:  # Show last 3 years of historical data
                previous_year = str(year_int - 1)
                previous_cost = historical_data.get(previous_year)
                change = ((rent - previous_cost) / previous_cost * 100) if previous_cost else 0
                
                recent_predictions.append({
                    'year': year,
                    'cost': rent,
                    'change': change,
                    'is_prediction': False
                })
        
        # Add predicted data
        for year, rent in predicted_data.items():
            year_int = int(year)
            previous_year = str(year_int - 1)
            previous_cost = predicted_data.get(previous_year) or historical_data.get(previous_year)
            
            if previous_cost:
                change = ((rent - previous_cost) / previous_cost * 100)
            else:
                change = 0
                
            recent_predictions.append({
                'year': year,
                'cost': rent,
                'change': change,
                'is_prediction': True
            })
        
        # Sort predictions by year
        recent_predictions.sort(key=lambda x: int(x['year']))
        
        context = {
            'chart_data': json.dumps(chart_data),
            'recent_predictions': recent_predictions
        }
        
        return render(request, 'predictor/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f"Error preparing dashboard: {str(e)}")
        print(f"Error details: {str(e)}")
        return render(request, 'predictor/dashboard.html', {
            'chart_data': json.dumps({'historical': {}, 'predicted': {}}),
            'recent_predictions': []
        })

@login_required
def change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')

        if not request.user.check_password(old_password):
            messages.error(request, 'Your current password is incorrect.')
            return redirect('profile')

        if new_password1 != new_password2:
            messages.error(request, 'New passwords do not match.')
            return redirect('profile')

        if len(new_password1) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return redirect('profile')

        if not any(char.isdigit() for char in new_password1) or not any(char.isalpha() for char in new_password1):
            messages.error(request, 'Password must contain both letters and numbers.')
            return redirect('profile')

        request.user.set_password(new_password1)
        request.user.save()
        messages.success(request, 'Your password has been changed successfully.')
        return redirect('profile')

    return redirect('profile')

