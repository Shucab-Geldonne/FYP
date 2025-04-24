import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def calculate_trends(data):
    """Calculate average yearly changes from historical data"""
    # Calculate year-over-year changes
    data['rent_change'] = data['Average Monthly Rent (England) (£)'].pct_change()
    data['petrol_change'] = data['Petrol Price (£/litre)'].pct_change()
    data['food_change'] = data['Weekly Food Expenditure (£)'].pct_change()
    
    # Calculate average changes over the last 5 years
    recent_data = data.tail(10)
    avg_rent_change = recent_data['rent_change'].mean()
    avg_petrol_change = recent_data['petrol_change'].mean()
    avg_food_change = recent_data['food_change'].mean()
    
    return {
        'rent_change': avg_rent_change,
        'petrol_change': avg_petrol_change,
        'food_change': avg_food_change
    }

def train_model(csv_path):
    """Train linear regression models for each cost type"""
    try:
        # Read and prepare data
        df = pd.read_csv(csv_path)
        df = df[['Year', 'Average Monthly Rent (England) (£)', 'Petrol Price (£/litre)', 'Weekly Food Expenditure (£)']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df = df.sort_values('Year')
        
        # Prepare features and targets
        X = df[['Year']].values
        y_rent = df['Average Monthly Rent (England) (£)'].values
        y_petrol = df['Petrol Price (£/litre)'].values
        y_food = df['Weekly Food Expenditure (£)'].values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        rent_model = LinearRegression()
        petrol_model = LinearRegression()
        food_model = LinearRegression()
        
        rent_model.fit(X_scaled, y_rent)
        petrol_model.fit(X_scaled, y_petrol)
        food_model.fit(X_scaled, y_food)
        
        return {
            'models': {
                'rent': rent_model,
                'petrol': petrol_model,
                'food': food_model
            },
            'scaler': scaler,
            'latest_year': df['Year'].max(),
            'latest_values': {
                'rent': df['Average Monthly Rent (England) (£)'].iloc[-1],
                'petrol': df['Petrol Price (£/litre)'].iloc[-1],
                'food': df['Weekly Food Expenditure (£)'].iloc[-1]
            }
        }
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return None

def predict(year, model_data):
    """Make predictions using the trained linear regression models"""
    try:
        if model_data is None:
            raise Exception("No trained models available")
        
        # Prepare input
        X = np.array([[year]])
        X_scaled = model_data['scaler'].transform(X)
        
        # Make predictions
        rent_pred = model_data['models']['rent'].predict(X_scaled)[0]
        petrol_pred = model_data['models']['petrol'].predict(X_scaled)[0]
        food_pred = model_data['models']['food'].predict(X_scaled)[0]
        
        # Get latest values for bounds checking
        latest_rent = model_data['latest_values']['rent']
        latest_petrol = model_data['latest_values']['petrol']
        latest_food = model_data['latest_values']['food']
        
        # Ensure predictions are reasonable
        rent_pred = max(rent_pred, latest_rent * 0.9)  # Don't predict below 90% of latest rent
        petrol_pred = max(petrol_pred, latest_petrol * 0.8)  # Don't predict below 80% of latest petrol
        food_pred = max(food_pred, latest_food * 0.8)  # Don't predict below 80% of latest food
        
        return {
            'rent': round(float(rent_pred), 2),
            'petrol': round(float(petrol_pred), 2),
            'food': round(float(food_pred), 2)
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None 