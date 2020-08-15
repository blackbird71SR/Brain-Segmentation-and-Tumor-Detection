from django import forms
from input.models import InputImage

class InputImageForm(forms.Form):
    image = forms.FileField(label="Upload a .zip File")