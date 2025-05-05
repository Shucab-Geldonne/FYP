@echo off
echo Setting up the project...

REM Create and activate virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment
    exit /b 1
)

REM Upgrade pip, setuptools and wheel
echo Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Failed to upgrade pip, setuptools and wheel
    exit /b 1
)

REM Install packages one by one with verification
echo Installing Django...
pip install Django==5.0.2
if errorlevel 1 (
    echo Failed to install Django
    exit /b 1
)

echo Installing numpy...
pip install numpy==1.26.4
if errorlevel 1 (
    echo Failed to install numpy
    exit /b 1
)

echo Installing pandas...
pip install pandas==2.2.1
if errorlevel 1 (
    echo Failed to install pandas
    exit /b 1
)

echo Installing scikit-learn...
pip install scikit-learn==1.4.1.post1
if errorlevel 1 (
    echo Failed to install scikit-learn
    exit /b 1
)

echo Installing django-crispy-forms...
pip install django-crispy-forms==2.1
if errorlevel 1 (
    echo Failed to install django-crispy-forms
    exit /b 1
)

echo Installing crispy-bootstrap4...
pip install crispy-bootstrap4==2023.1
if errorlevel 1 (
    echo Failed to install crispy-bootstrap4
    exit /b 1
)

REM Verify Django is installed
echo Verifying Django installation...
python -c "import django; print('Django version:', django.get_version())"
if errorlevel 1 (
    echo Django installation verification failed
    exit /b 1
)

REM Create optimized requirements file
echo Creating optimized requirements file...
pip freeze > requirements_optimized.txt
if errorlevel 1 (
    echo Failed to create requirements file
    exit /b 1
)

REM Run migrations
echo Running migrations...
python manage.py makemigrations
if errorlevel 1 (
    echo Failed to create migrations
    exit /b 1
)

python manage.py migrate
if errorlevel 1 (
    echo Failed to apply migrations
    exit /b 1
)

REM Create superuser if it doesn't exist
echo Checking for superuser...
python manage.py shell -c "from django.contrib.auth import get_user_model; User = get_user_model(); print('Superuser already exists' if User.objects.filter(username='admin').exists() else (User.objects.create_superuser('admin', 'admin@example.com', 'admin') and print('Superuser created')))"
if errorlevel 1 (
    echo Failed to create superuser
    exit /b 1
)

REM Start the development server
echo Starting development server...
python manage.py runserver 