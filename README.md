# Cost of Living Predictor

A Django web application that predicts future cost of living changes based on historical data.

## Installation

### For Windows Users:

1. Clone the repository
2. Run the setup script:
   ```bash
   .\setup_and_test.bat
   ```

### For Mac Users:

1. Clone the repository
2. Make the setup script executable:
   ```bash
   chmod +x setup_and_test.sh
   ```
3. Run the setup script:
   ```bash
   ./setup_and_test.sh
   ```

The setup script will:

- Create a virtual environment
- Install all required dependencies
- Run database migrations
- Create a default admin user (username: admin, password: admin)
- Start the development server

The application will be available at: http://127.0.0.1:8000/

## Features

- Cost prediction based on historical data
- User authentication and authorization
- Interactive dashboard
- Event calendar
- Cost tracking and analysis

## Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (created automatically by the setup script)

## Project Structure

- `predictor/` - Main application directory
  - `models.py` - Database models
  - `views.py` - View functions and classes
  - `urls.py` - URL routing
  - `templates/` - HTML templates
  - `static/` - Static files (CSS, JavaScript, images)
- `cost_living_predictor/` - Project settings
- `requirements.txt` - Python dependencies
- `setup_and_test.bat` - Windows setup script
- `setup_and_test.sh` - Mac setup script

## License

This project is licensed under the MIT License - see the LICENSE file for details.
