import unittest
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils import timezone
from .models import Event, Prediction
from .simple_model import calculate_trends, train_model, predict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SimpleModelTests(TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Year': [2018, 2019, 2020, 2021, 2022],
            'Average Monthly Rent (England) (£)': [800, 850, 900, 950, 1000],
            'Petrol Price (£/litre)': [1.2, 1.3, 1.4, 1.5, 1.6],
            'Weekly Food Expenditure (£)': [50, 55, 60, 65, 70]
        })
        self.sample_data.to_csv('test_data.csv', index=False)
        self.model_data = train_model('test_data.csv')

    def test_calculate_trends(self):
        trends = calculate_trends(self.sample_data)
        self.assertIsInstance(trends, dict)
        self.assertIn('rent_change', trends)
        self.assertIn('petrol_change', trends)
        self.assertIn('food_change', trends)

    def test_train_model(self):
        self.assertIsNotNone(self.model_data)
        self.assertIn('models', self.model_data)
        self.assertIn('scaler', self.model_data)
        self.assertIn('latest_year', self.model_data)
        self.assertIn('latest_values', self.model_data)

    def test_predict(self):
        predictions = predict(2023, self.model_data)
        self.assertIsNotNone(predictions)
        self.assertIn('rent', predictions)
        self.assertIn('petrol', predictions)
        self.assertIn('food', predictions)
        self.assertGreater(predictions['rent'], 0)
        self.assertGreater(predictions['petrol'], 0)
        self.assertGreater(predictions['food'], 0)

class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_home_view_redirect_if_not_logged_in(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 302)  # Redirect to login page
        self.assertTrue(response.url.startswith('/accounts/login/'))

    def test_home_view_logged_in(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/home.html')

    def test_predictions_view_redirect_if_not_logged_in(self):
        response = self.client.get(reverse('predictions'))
        self.assertEqual(response.status_code, 302)  # Redirect to login page
        self.assertTrue(response.url.startswith('/accounts/login/'))

    def test_predictions_view_logged_in(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('predictions'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/predictions.html')

class EventModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.event = Event.objects.create(
            title='Test Event',
            description='Test Description',
            date=timezone.now().date()
        )

    def test_event_creation(self):
        self.assertEqual(self.event.title, 'Test Event')
        self.assertEqual(self.event.description, 'Test Description')
        self.assertIsNotNone(self.event.date)

    def test_event_str_representation(self):
        self.assertEqual(str(self.event), 'Test Event')

class PredictionModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.prediction = Prediction.objects.create(
            user=self.user,
            date=timezone.now().date(),
            rent_prediction=1200.00,
            petrol_prediction=1.75,
            food_prediction=75.00
        )

    def test_prediction_creation(self):
        self.assertEqual(self.prediction.user, self.user)
        self.assertIsNotNone(self.prediction.date)
        self.assertEqual(float(self.prediction.rent_prediction), 1200.00)
        self.assertEqual(float(self.prediction.petrol_prediction), 1.75)
        self.assertEqual(float(self.prediction.food_prediction), 75.00)

    def test_prediction_str_representation(self):
        expected_str = f"Prediction for {self.prediction.date} by {self.user.username}"
        self.assertEqual(str(self.prediction), expected_str)

class AuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.login_url = '/accounts/login/'  # Django's default login URL
        self.logout_url = '/accounts/logout/'  # Django's default logout URL

    def test_user_login(self):
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'testpass'
        })
        self.assertEqual(response.status_code, 302)  # Redirect after successful login

    def test_user_logout(self):
        # First login
        self.client.login(username='testuser', password='testpass')
        # Then logout using POST
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 302)  # Redirect after logout

class RegistrationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = '/accounts/register/'
        self.login_url = '/accounts/login/'

    def test_registration_page(self):
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/register.html')

    def test_successful_registration(self):
        # Print initial user count
        print(f"\nUsers before registration: {User.objects.all().count()}")
        print("Existing usernames:", [user.username for user in User.objects.all()])
        
        response = self.client.post(self.register_url, {
            'username': 'newuser',
            'password1': 'complexpassword123',
            'password2': 'complexpassword123',
            'email': 'newuser@example.com'
        })
        
        # Print final user count and list all users
        print(f"\nUsers after registration: {User.objects.all().count()}")
        print("Final usernames:", [user.username for user in User.objects.all()])
        
        self.assertEqual(response.status_code, 302)  # Redirect after successful registration
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_registration_with_existing_username(self):
        # Create a user first
        User.objects.create_user(username='existinguser', password='testpass')
        
        response = self.client.post(self.register_url, {
            'username': 'existinguser',  # Try to register with same username
            'password1': 'complexpassword123',
            'password2': 'complexpassword123',
            'email': 'newuser@example.com'
        })
        self.assertEqual(response.status_code, 200)  # Should stay on registration page
        self.assertContains(response, 'A user with that username already exists')

    def test_registration_with_mismatched_passwords(self):
        response = self.client.post(self.register_url, {
            'username': 'newuser',
            'password1': 'complexpassword123',
            'password2': 'differentpassword',  # Mismatched passwords
            'email': 'newuser@example.com'
        })
        self.assertEqual(response.status_code, 200)  # Should stay on registration page
        self.assertContains(response, 'The two password fields didn\u2019t match.')

if __name__ == '__main__':
    unittest.main() 