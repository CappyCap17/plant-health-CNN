🌿 Plant Health Classification Web App
A deep learning-powered web platform to detect plant diseases using images of plant leaves. Built with a Convolutional Neural Network (CNN) and deployed via a Django web interface,
 this tool is designed to help farmers and agricultural specialists quickly diagnose plant health issues and receive treatment suggestions.

🧠 Project Abstract
India's agriculture, crucial for food security, suffers from delayed and inaccurate plant disease diagnosis.
 This project introduces a user-friendly web app that allows users to upload plant images and instantly detect diseases using a trained CNN model. 
The goal is to assist in timely diagnosis and reduce crop losses.

🚀 Features
1. Upload plant leaf images via phone or desktop.

2. Automatic disease detection using a trained CNN model.

3. Real-time treatment recommendations via a connected JSON file.

4. Optimized for usability across devices.

5. Achieved 93% accuracy with augmented training data.

🛠️ Tech Stack
Frontend: Django templates, HTML, CSS

Backend: Django (Python)

Model: Custom CNN with Keras

Dataset: Augmented version of PlantVillage (87,000+ images, 38 classes)

Deployment: Local Django server or production-ready via services like Heroku, Render, etc.

🧬 CNN Model Architecture
Input Layer: 250x250x3 RGB images

3× Convolutional + MaxPooling2D layers

Flatten layer

Dense Layer with 128 ReLU neurons

Dropout Layer (0.5)

Output Layer: SoftMax for 3 classes — Healthy, Powdery Mildew, Rust

🧪 Training & Performance
Epochs: 20 (with early stopping)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy: 93%

Evaluation: Confusion Matrix, Precision, Recall, F1-Score

Category	Precision	Recall	F1-Score
Healthy  	0.85	    0.82	  0.84
Powdery	  0.82	    0.90	  0.86
Rust	    1.00	    0.94	  0.97

🗃️ Dataset Augmentation Techniques
To increase generalization and robustness:

Rotation (±40°)

Width/Height shift (20%)

Shear, Zoom

Horizontal flip

Color jittering (brightness, saturation, hue)

Normalization (rescale to [0, 1])

📸 Example Inputs
Healthy leaf

Powdery mildew-infected leaf

Rust-infected leaf

(Images are part of the enhanced dataset)

🌐 User Workflow
Visit the web app.

Upload an image of a plant leaf.

Choose plant type (from 50+ common Indian plants).

Receive instant disease classification + treatment recommendation.

🔄 Future Work(If i get time lol)
Integrate multilingual support for Indian farmers.

API-based treatment recommendation from agricultural databases.

Model improvement with more diverse and noisy datasets.

Deploy as a cross-platform mobile app.

📂 Project Structure (Sample)
java
Copy
Edit
project-root/
│
├── image_app_project/
│   ├── templates/
│   ├── static/
│   ├── models/        ← Trained CNN model (non-PKL format)
│   ├── views.py
│   ├── urls.py
│   └── ...
├── manage.py
└── README.md
🤝 Target Audience
People who love gardening, into botanical stuff, Farmers and rural agronomists

Agricultural researchers

Students and developers interested in AI in agriculture

 These can find it helpful and if you into CS and AI projects, you can use this to make further improvements!

📜 License
This project is open-source and free to use for educational and non-commercial purposes but do acknowledge my work lol!.
 

(Psst: I used chatgpt and perplexity ai to make this ReadMe and project is made by my logic + ai alone for my diploma final year project as team members were lazy
and dependent on me)

Will come with more interesting projects in future,
Contributions are welcomed and do so follow my linkedIn and Github to connect!
Cya!
