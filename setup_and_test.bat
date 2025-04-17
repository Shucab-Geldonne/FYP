@echo off
echo Setting up the project...

REM Create and activate virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

REM Install only core packages
echo Installing core packages...
pip install Django==5.0.2 numpy>=1.21.0,<2.0.0 pandas>=2.1.4 scikit-learn>=1.3.2 django-crispy-forms==2.1 crispy-bootstrap4==2023.1

REM Create optimized requirements file
echo Creating optimized requirements file...
pip freeze > requirements_optimized.txt

REM Run migrations
echo Running migrations...
python manage.py makemigrations
python manage.py migrate

REM Create superuser if it doesn't exist
echo Checking for superuser...
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print('Superuser created')
else:
    print('Superuser already exists')
"

REM Start the development server
echo Starting development server...
python manage.py runserver 