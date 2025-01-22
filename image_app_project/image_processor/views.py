
import json
import os
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
import tensorflow as tf
import numpy as np

# # def upload_image(request):
# #     # Load JSON data
# #     data_path = os.path.join(settings.BASE_DIR, 'data/recommendations.json')  # Ensure the path is correct
# #     with open(data_path, 'r') as file:
# #         recommendations = json.load(file)
    
# #     # Extract plant names for the dropdown
# #     plants = {item["plant_name"] for item in recommendations}

# #     if request.method == 'POST':
# #         form = ImageUploadForm(request.POST, request.FILES)
# #         if form.is_valid():
# #             uploaded_file = form.save()
# #             image_path = uploaded_file.image.path
# #             result, _ = predict_image(image_path)

# #             # Get plant name (either from the dropdown or manual input)
# #             plant_name = request.POST.get('plant_name') or request.POST.get('plant_name_manual', 'Not provided')

# #             # Fetch recommendations for the predicted disease
# #             treatment = next(
# #                 (item["recommended_treatment"] for item in recommendations if item["plant_name"] == plant_name and item["disease_type"].lower() == result.lower()),
# #                 "No treatment available."
# #             )

# #             return render(request, 'image_processor/result.html', {
# #                 'result': result,
# #                 'plant_name': plant_name,
# #                 'treatment_recommendations': treatment,
# #             })
# #     else:
# #         form = ImageUploadForm()

# #     return render(request, 'image_processor/upload.html', {'form': form, 'plants': plants})

# def upload_image(request):
#     # Load JSON data
#     data_path = os.path.join(settings.BASE_DIR, 'data/recommendations.json')
#     with open(data_path, 'r') as file:
#         recommendations = json.load(file)

#     # Extract plant names for the dropdown
#     plants = {item["plant_name"] for item in recommendations}

#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_file = form.save()
#             image_path = uploaded_file.image.path
#             result, predictions = predict_image(image_path)

#             # Get plant name from dropdown
#             plant_name = request.POST.get('plant_name')

#             if result == "Healthy":
#                 return render(request, 'result.html', {
#                     'plant_name': plant_name,
#                     'result': result,
#                     'accuracy': round(np.max(predictions) * 100, 2),
#                 })

#             # Fetch treatment recommendations
#             matched_recommendation = next(
#                 (item for item in recommendations if item["plant_name"] == plant_name and item["disease_type"].lower() == result.lower()), 
#                 None
#             )

#             return render(request, 'result.html', {
#                 'plant_name': plant_name,
#                 'result': result,
#                 'symptoms': matched_recommendation["symptoms"] if matched_recommendation else "N/A",
#                 'treatment_recommendations': matched_recommendation["recommended_treatment"] if matched_recommendation else "No treatment available.",
#                 'additional_notes': matched_recommendation["additional_notes"] if matched_recommendation else "No additional notes.",
#                 'accuracy': round(np.max(predictions) * 100, 2),
#             })

#     else:
#         form = ImageUploadForm()

#     return render(request, 'upload.html', {'form': form, 'plants': plants})


# def predict_image(image_path):
#     # Load the trained model
#     model_path = os.path.join(settings.BASE_DIR, 'models/plant_health_classifier.h5')
#     model = tf.keras.models.load_model(model_path)

#     # Prepare the image for prediction
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(250, 250))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict the disease
#     predictions = model.predict(img_array)
#     class_labels = ['Healthy', 'Powdery', 'Rust']  # Example classes
#     return class_labels[np.argmax(predictions)], predictions[0]

# views.py
import json
import os
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
import tensorflow as tf
import numpy as np

def upload_image(request):
    # Load JSON data
    data_path = os.path.join(settings.BASE_DIR, 'data/recommendations.json')
    with open(data_path, 'r') as file:
        recommendations = json.load(file)

    # Extract plant names for the dropdown
    plants = {item["plant_name"] for item in recommendations}

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            image_path = uploaded_file.image.path
            result, accuracy = predict_image(image_path)

            plant_name = request.POST.get('plant_name', 'Not provided')

            # Fetch recommendations for the predicted disease
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
    # Load the trained model
    model_path = os.path.join(settings.BASE_DIR, 'models/plant_health_classifier.h5')
    model = tf.keras.models.load_model(model_path)

    # Prepare the image for prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(250, 250))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the disease
    predictions = model.predict(img_array)
    class_labels = ['Healthy', 'Powdery', 'Rust']  # Example classes
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    accuracy = predictions[0][predicted_class] * 100  # Convert to percentage

    return predicted_label, accuracy
