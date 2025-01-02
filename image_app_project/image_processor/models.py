# from django.db import models

# class UploadedImage(models.Model):
#     image = models.ImageField(upload_to='images/')
#     uploaded_at = models.DateTimeField(auto_now_add=True)
from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')  # The uploaded image is stored in 'images/' folder
    uploaded_at = models.DateTimeField(auto_now_add=True)
