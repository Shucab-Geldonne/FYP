# Cost of Living Prediction System

This project uses machine learning (PyTorch) to predict cost of living based on various factors. The system is built with Django for the web interface and PyTorch for the machine learning components.

## Features

- Cost of living prediction using machine learning
- User-friendly web interface
- Data visualization and analysis
- Interactive prediction form

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:

```bash
venv\Scripts\activate
```

- Unix/MacOS:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run migrations:

```bash
python manage.py makemigrations
python manage.py migrate
```

5. Start the development server:

```bash
python manage.py runserver
```

## Project Structure

- `cost_living_predictor/` - Main Django project directory
- `ml_model/` - Machine learning model and related components
- `predictor/` - Django app for handling predictions
- `data/` - Directory for dataset storage

## Technology Stack

- Django 5.0.2
- PyTorch 2.2.0
- Python 3.x
- scikit-learn
- pandas
- numpy
