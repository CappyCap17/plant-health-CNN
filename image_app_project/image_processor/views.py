import json
import os
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
import tensorflow as tf
import numpy as np

def upload_image(request):
  
    data_path = os.path.join(settings.BASE_DIR, 'data/recommendations.json')
    with open(data_path, 'r') as file:
        recommendations = json.load(file)

   
    plants = {item["plant_name"] for item in recommendations}

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            image_path = uploaded_file.image.path
            result, accuracy = predict_image(image_path)

            plant_name = request.POST.get('plant_name', 'Not provided')

           
            treatment = next(
                (item for item in recommendations if item["plant_name"] == plant_name and item["disease_type"].lower() == result.lower()),
                None
            )

            context = {
                'result': result,
                'accuracy': accuracy,
                'plant_name': plant_name,
                'treatment_recommendations': treatment["recommended_treatment"] if treatment else "No treatment available.",
                'additional_notes': treatment["additional_notes"] if treatment else "No additional notes available.",
                'symptoms': treatment["symptoms"] if treatment else "No symptoms available.",
            }
            return render(request, 'result.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form, 'plants': plants})

def predict_image(image_path):
   
    model_path = os.path.join(settings.BASE_DIR, 'models/plant_health_classifier.h5')
    model = tf.keras.models.load_model(model_path)


    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(250, 250))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    
    predictions = model.predict(img_array)
    class_labels = ['Healthy', 'Powdery', 'Rust']  
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    accuracy = predictions[0][predicted_class] * 100  

    return predicted_label, accuracy
