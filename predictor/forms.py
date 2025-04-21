from django import forms
from .models import CostOfLivingData
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Field

class CostOfLivingForm(forms.ModelForm):
    class Meta:
        model = CostOfLivingData
        fields = ['city', 'country', 'rent', 'groceries', 'utilities', 'transportation', 'income']
        widgets = {
            'city': forms.TextInput(attrs={'class': 'form-control'}),
            'country': forms.TextInput(attrs={'class': 'form-control'}),
            'rent': forms.NumberInput(attrs={'class': 'form-control'}),
            'groceries': forms.NumberInput(attrs={'class': 'form-control'}),
            'utilities': forms.NumberInput(attrs={'class': 'form-control'}),
            'transportation': forms.NumberInput(attrs={'class': 'form-control'}),
            'income': forms.NumberInput(attrs={'class': 'form-control'})
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))
        self.helper.layout = Layout(
            Field('city', css_class='form-control'),
            Field('country', css_class='form-control'),
            Field('rent', css_class='form-control'),
            Field('groceries', css_class='form-control'),
            Field('utilities', css_class='form-control'),
            Field('transportation', css_class='form-control'),
            Field('income', css_class='form-control'),
        ) 