# Cost of Living Predictor

A Django web application that predicts future cost of living changes based on historical data.

## Quick Start

1. **Install Python 3.12**

   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Set up the project**

   ```bash
   # Create and activate virtual environment
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux

   # Install dependencies
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

   # Run migrations and start server
   python manage.py migrate
   python manage.py runserver
   ```

3. **Access the application**
   - Open your browser and go to `http://127.0.0.1:8000/`

## Features

- Cost prediction based on historical data
- User authentication and authorization
- Interactive dashboard
- Event calendar
- Cost tracking and analysis

## Troubleshooting

If you encounter any issues:

1. **Python not found**

   - Make sure Python is added to your PATH
   - Try using `python -m pip` instead of just `pip`

2. **Installation errors**

   ```bash
   # Upgrade pip and build tools
   python -m pip install --upgrade pip setuptools wheel

   # Try installing requirements again
   pip install -r requirements.txt
   ```

3. **SSL errors**
   ```bash
   python -m pip install --upgrade certifi
   ```

## Project Structure

```
predictor/              # Main application
├── models.py          # Database models
├── views.py           # View functions
├── urls.py            # URL routing
├── templates/         # HTML templates
└── static/           # CSS, JavaScript, images
```

## Requirements

- Python 3.12
- pip (Python package installer)
- Virtual environment

## License

This project is licensed under the MIT License - see the LICENSE file for details.
