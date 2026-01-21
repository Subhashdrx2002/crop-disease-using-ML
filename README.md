# Crop Disease Detection Using Deep Learning

Final Year B.Tech Project (BTP) — Netaji Subhas University of Technology (NSUT), Delhi

📌 Project Overview

Crop diseases are a significant threat to food security, causing ~42% annual agricultural loss globally. Traditional diagnosis depends on visual inspection by experts — a slow, subjective, and non-scalable method.

This project builds an automated crop disease detection system using deep learning to classify leaf images into disease categories. The goal is to enable faster diagnosis, support precision agriculture, and assist farmers in preventing large-scale crop loss.

🎯 Objectives

✔ Identify plant diseases from leaf images using ML
✔ Benchmark multiple architectures under identical conditions
✔ Improve accuracy while reducing computational overhead
✔ Explore feasibility for mobile/edge deployment in agriculture
✔ Support automation & accessibility for rural usage

🧠 Model Architectures

This project implements and benchmarks three CNN-based architectures:

Model	Type	Accuracy
CNN (Baseline)	Custom	27.36%
ResNet-50	Transfer Learning	61.21%
EfficientNet-B3	Transfer Learning + Scaling	~95%

EfficientNet-B3 achieved the best diagnostic performance while being computationally efficient (Paper Result) 

Conference_Paper_BTP

🧰 Tech Stack

Languages & Libraries

Python

TensorFlow / Keras

NumPy

OpenCV

Techniques

Convolutional Neural Networks (CNN)

Transfer Learning (ResNet50, EfficientNetB3)

Feature Extraction

Fine-Tuning

Data Augmentation

Tools

Google Colab

Jupyter Notebook

📊 Dataset

Dataset includes multi-species leaf images with:

Healthy crops

Infected crops (multiple disease classes)

Dataset split:

70–80% Training

10–15% Validation

10–15% Testing

Image augmentations applied to improve generalization:

Rotation

Zoom

Flip

Shift

⚙️ System Workflow

Workflow (Presentation Page 9) 

Final End Semester BTP Present…

Input Image → Preprocessing → Feature Extraction → Classification → Disease Output

🧪 Results & Evaluation
📌 Performance Comparison
Model	Accuracy
CNN	27.36%
ResNet50	61.21%
EfficientNetB3	94.93% – 95%
📌 Key Insight

Progressively deeper and scaled architectures significantly improved diagnostic capability.

🌱 Real-World Applications

✔ Precision Agriculture
✔ Smart Farming Systems
✔ Crop Advisory Platforms
✔ Mobile Disease Diagnostic Apps
✔ Yield Optimization & Decision Support

🔍 Why This Matters

Agriculture is still largely diagnosis-dependent; delays allow diseases to spread and reduce yields.

This system offers:
✔ Early intervention
✔ Reduced dependency on experts
✔ Faster decision-making
✔ Increased accessibility in rural regions

🚀 How to Run
Clone Repo
git clone https://github.com/<username>/crop-disease-detection.git
cd crop-disease-detection

Install Dependencies
pip install -r requirements.txt

Predict on a New Image
python predict.py --image sample_leaf.jpg

🏗 Future Work

Planned improvements:

Mobile deployment (TensorFlow Lite)

Edge computing deployment (Jetson Nano / Raspberry Pi)

Disease severity estimation (not just classification)

Multi-crop dataset scaling

Recommendation system for treatment/pesticides

Real-world field testing

📄 Research Paper & Documentation

📄 Conference Paper (PDF)

Crop Disease Detection Using Machine Learning-Based Image Classification 

Conference_Paper_BTP

📊 BTP Presentation (Final Viva)

Crop Disease Detection Using ML (24 Slides) 

Final End Semester BTP Present…

👩‍🌾 Domain Impact (AgriTech)

This project supports:
✔ Food Security
✔ Sustainable Agriculture
✔ Farmer Assistance
✔ Climate-Resilient Farming

🧑‍💻 Team

Subhash (DL + Model Development)

Yashika Kumar

Anushka Nimi

Vishal Tomar

Department of Instrumentation & Control Engineering (ICE)
NSUT, Delhi

📬 Contact

For queries or collaboration:
📧 subhash.ug22@nsut.ac.in

🔗 GitHub: https://github.com/Subhashdrx2002

🔗 LinkedIn: https://www.linkedin.com/in/subhash-kumar-782513257/
