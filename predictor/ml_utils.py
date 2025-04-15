import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class CostPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

class CostPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.columns = None
    
    def prepare_data(self, csv_path):
        # Read and prepare the data
        df = pd.read_csv(csv_path)
        print("\nTraining Data Shape:", df.shape)
        print("\nColumns being predicted:", df.columns[1:].tolist())
        self.columns = df.columns[1:]  # Exclude 'Year' column
        
        # Prepare features (X) and targets (y)
        X = df[['Year']].values
        y = df[self.columns].values
        
        print("\nInput years:", X.flatten().tolist())
        
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def train_model(self, csv_path, epochs=1000):
        print("\nStarting model training...")
        X_scaled, y_scaled = self.prepare_data(csv_path)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Initialize model with correct output size
        output_size = len(self.columns)
        print(f"\nModel architecture: Input(1) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(32) -> ReLU -> Dropout(0.2) -> Linear({output_size})")
        self.model = CostPredictionModel(input_size=1, output_size=output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training loop
        self.model.train()
        print("\nTraining Progress:")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        self.model.eval()
        print("\nTraining completed!")
        
        # Show sample prediction
        with torch.no_grad():
            sample_year = 2022
            sample_input = torch.FloatTensor(self.scaler_X.transform([[sample_year]]))
            sample_output = self.model(sample_input)
            sample_pred = self.scaler_y.inverse_transform(sample_output)
            print(f"\nSample prediction for {sample_year}:")
            for i, col in enumerate(self.columns):
                print(f"{col}: £{sample_pred[0][i]:.2f}")
    
    def predict(self, year):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input
        X_pred = np.array([[year]])
        X_pred_scaled = self.scaler_X.transform(X_pred)
        X_pred_tensor = torch.FloatTensor(X_pred_scaled)
        
        # Make prediction
        with torch.no_grad():
            y_pred_scaled = self.model(X_pred_tensor)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Create prediction dictionary
        predictions = {}
        for i, col in enumerate(self.columns):
            predictions[col] = round(float(y_pred[0][i]), 2)
        
        return predictions
    
    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'columns': self.columns
        }
        torch.save(model_state, path)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Model file not found at {path}")
        
        model_state = torch.load(path)
        self.columns = model_state['columns']
        self.scaler_X = model_state['scaler_X']
        self.scaler_y = model_state['scaler_y']
        
        # Initialize model with correct output size
        output_size = len(self.columns)
        self.model = CostPredictionModel(input_size=1, output_size=output_size)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.eval()
        print(f"\nModel loaded from {path}")
        print(f"Number of features: {output_size}")
        print("Features:", self.columns.tolist()) 