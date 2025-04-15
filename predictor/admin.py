from django.contrib import admin
from .models import CostOfLivingData

@admin.register(CostOfLivingData)
class CostOfLivingDataAdmin(admin.ModelAdmin):
    list_display = ('city', 'country', 'rent', 'groceries', 'utilities', 'transportation', 'income', 'timestamp')
    list_filter = ('country', 'timestamp')
    search_fields = ('city', 'country')
