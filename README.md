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

1. **Prerequisites**:

   - Install Python 3.x (if not already installed)
   - Install pip (Python package manager)

2. **Setup Steps**:

   a. **Clone the repository**:

   ```bash
   git clone https://github.com/Shucab-Geldonne/FYP.git
   cd finalYearProject4
   ```

   b. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   c. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   d. **Run migrations**:

   ```bash
   python manage.py migrate
   ```

   e. **Start the development server**:

   ```bash
   python manage.py runserver
   ```

3. **Access the application**:
   - Open a web browser and go to `http://127.0.0.1:8000/`

**Troubleshooting Tips for Mac Users**:

1. If you encounter permission issues, you might need to use `sudo` for some commands
2. If you get any SSL-related errors, you might need to install the SSL certificates:
   ```bash
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```
3. If you have multiple Python versions installed, make sure to use `python3` explicitly

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
