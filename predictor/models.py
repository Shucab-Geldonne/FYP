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
