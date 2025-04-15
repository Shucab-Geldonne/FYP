@echo off
echo Starting project setup and testing...

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies
    exit /b 1
)

:: Run migrations
echo Running migrations...
python manage.py makemigrations
if %ERRORLEVEL% neq 0 (
    echo Failed to create migrations
    exit /b 1
)

python manage.py migrate
if %ERRORLEVEL% neq 0 (
    echo Failed to apply migrations
    exit /b 1
)

:: Start the development server
echo Starting development server...
echo The server will start on http://127.0.0.1:8000/
echo Press Ctrl+C to stop the server
python manage.py runserver

:: Deactivate virtual environment when done
call venv\Scripts\deactivate 