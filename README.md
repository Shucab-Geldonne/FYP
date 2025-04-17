# Cost of Living Predictor

A Django web application that predicts future costs of living (rent, petrol, and food) using linear regression models based on historical data.

## Features

- User authentication and authorization
- Historical cost data visualization
- Future cost predictions using linear regression
- Interactive dashboard with charts and trends
- Event management system
- User profile management

## Technical Stack

- **Backend**: Django 5.0.2
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: scikit-learn (Linear Regression)
- **Frontend**: Bootstrap 4, Chart.js
- **Database**: SQLite (development)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd FYP
```

2. Run the setup script:

```bash
# On Windows
.\setup_and_test.bat
```

The setup script will:

- Create and activate a virtual environment
- Install all required dependencies
- Run database migrations
- Create a default admin user (username: admin, password: admin)
- Start the development server

Once the server starts, you can access the application at http://127.0.0.1:8000/

## Project Structure

- `cost_living_predictor/`: Main project configuration
- `predictor/`: Main application
  - `models.py`: Database models
  - `views.py`: View logic and prediction handling
  - `simple_model.py`: Linear regression prediction models
  - `templates/`: HTML templates
  - `static/`: CSS, JavaScript, and other static files
- `accounts/`: User authentication and profile management
- `historical_costs.csv`: Historical data for training models

## Prediction Model

The application uses linear regression models to predict future costs:

- Separate models for rent, petrol, and food costs
- Trained on historical data from `historical_costs.csv`
- Predictions are made based on yearly trends
- Includes reasonable bounds to prevent unrealistic predictions

## Usage

1. Log in to the application
2. View historical data and trends on the dashboard
3. Make predictions for future years
4. Manage your profile and events
5. View and analyze cost trends

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
