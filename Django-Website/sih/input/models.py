import os
from django.db import models

# Create your models here.

class InputImage(models.Model):
	image = models.FileField(upload_to='input/')
	# image_gz = models.FileField()
 
	def extension(self):
		exten = os.path.splitext(self.image.name)
		return exten[1]