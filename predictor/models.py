from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.

class CostOfLivingData(models.Model):
    city = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    rent = models.FloatField(help_text="Average rent for 1 bedroom apartment")
    groceries = models.FloatField(help_text="Monthly groceries cost")
    utilities = models.FloatField(help_text="Monthly utilities cost")
    transportation = models.FloatField(help_text="Monthly transportation cost")
    income = models.FloatField(help_text="Average monthly income")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.city}, {self.country}"

    class Meta:
        verbose_name_plural = "Cost of Living Data"

class Event(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    color = models.CharField(max_length=20, default='#3788d8')  # Default blue color
    
    class Meta:
        ordering = ['start_date']
    
    def __str__(self):
        return self.title
    
    @property
    def is_upcoming(self):
        now = timezone.now()
        return (self.start_date > now and 
                (self.start_date - now).days <= 7)  # Events within next 7 days
