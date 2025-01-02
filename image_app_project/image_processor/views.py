# import os
# import pickle
# from django.shortcuts import render
# from django.conf import settings
# from django.core.files.storage import FileSystemStorage
# from PIL import Image
# from .forms import ImageUploadForm
# import numpy as np
# import tensorflow as tf

# def upload_image(request):
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_file = form.save()
#             image_path = uploaded_file.image.path

#             predicted_label, confidence_scores = predict_image(image_path)
#             result = predicted_label

#             return render(request, 'result.html', {'result': result})
#     else:
#         form = ImageUploadForm()
#     return render(request, 'upload.html', {'form': form})



# def predict_image(image_path):
#     # Load the saved model
    
#     # model_path = os.path.join(settings.BASE_DIR, 'models/model.pkl')
#     model_path = os.path.join(settings.BASE_DIR, 'models/plant_health_classifier.h5')
#     model = tf.keras.models.load_model(model_path)

#     # Preprocess the input image
#     img_width = 250
#     img_height = 250
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Predict the class
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     class_labels = list(['Healthy', 'Powdery', 'Rust'])  # Get class labels
#     predicted_class_label = class_labels[predicted_class_index]

#     return predicted_class_label, predictions[0]

import os
import json
from django.shortcuts import render
from .forms import ImageUploadForm
import tensorflow as tf
import numpy as np
import json
from django.conf import settings


def upload_image(request):
    # Load JSON data
    data_path = os.path.join(settings.BASE_DIR, 'data/recommendations.json')
    with open(data_path, 'r') as file:
        recommendations = json.load(file)
    
    plants = {item["plant_name"] for item in recommendations}

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            image_path = uploaded_file.image.path
            result, _ = predict_image(image_path)

            # Fetch recommendations
            plant_name = request.POST.get('plant_name') or request.POST.get('plant_name_manual', 'Not provided')
            treatment = next(
                (item["recommended_treatment"] for item in recommendations if item["plant_name"] == plant_name and item["disease_type"].lower() == result.lower()),
                "No treatment available."
            )

            return render(request, 'result.html', {
                'result': result,
                'plant_name': plant_name,
                'treatment_recommendations': treatment,
            })
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
    return class_labels[np.argmax(predictions)], predictions[0]


def get_treatment_recommendations(plant_name, disease):
    # Construct the path to the JSON file
    json_file_path = settings.DATA_ROOT / 'recommendations.json'

    # Load the JSON data
    with open(json_file_path, 'r') as file:
        recommendations = json.load(file)

    # Fetch the treatment recommendation for the plant and disease
    if plant_name in recommendations and disease in recommendations[plant_name]:
        return recommendations[plant_name][disease]
    else:
        return "No specific recommendations available."


