from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.

class Event(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    rent_prediction = models.DecimalField(max_digits=10, decimal_places=2)
    petrol_prediction = models.DecimalField(max_digits=10, decimal_places=2)
    food_prediction = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.date} by {self.user.username}"
