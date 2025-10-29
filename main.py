import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Global Configurations
# -------------------------------
MODEL_PATH = "best_flower_model.keras"
IMG_SIZE = (160, 160)

# Load model only once (cached)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Train and save best_flower_model.keras first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define class names for Flower Dataset
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Ensure correct ordering

# Prediction Function
def predict_image(img):
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    score = np.max(preds)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    return pred_class, score

# -------------------------------
# Streamlit Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About", "Flower Classification"])

if page == "Home":
    st.title("🌸 Welcome to the Intelligent Flower Recognition System")

    st.markdown("""
    ---
    ### 🌼 Project Overview
    This application is a **Deep Learning–powered Flower Classification System** that uses
    **Transfer Learning** with **MobileNetV2**, a neural network pretrained on the
    **ImageNet** dataset.

    Flowers are one of nature’s most diverse and visually appealing objects, yet
    distinguishing between similar flower species can be challenging for the human eye.
    Thanks to advancements in computer vision, deep learning can now **identify flower species
    with high accuracy**, making flower recognition useful in:

    ✅ Botanical Research  
    ✅ Agricultural Monitoring  
    ✅ Smart Gardening Apps  
    ✅ Environmental Studies  
    ✅ Educational Tools  

    ---

    ### 📌 Dataset Information
    The dataset used in this project is the official **Flower Photos Dataset** provided by
    **TensorFlow**. It contains images from 5 popular flower categories:

    - 🌼 **Daisy** — Small, bright flowers with white petals
    - 🌻 **Dandelion** — Famous puffball seed heads
    - 🌹 **Roses** — Elegant and diverse species symbolizing love
    - 🌞 **Sunflowers** — Large yellow blooms following sunlight
    - 🌷 **Tulips** — Colorful bell-shaped spring flowers

    Each image is different in:
    - Lighting conditions
    - Angles & background clutter
    - Flower color variations

    Making classification a fun challenge for AI! 😄

    ---

    ### 🧠 What Technology is Used?
    We compared multiple approaches:

    | Model Type | Description | Accuracy |
    |------------|-------------|---------|
    | Scratch CNN | Built from 0 layers | Good but limited |
    | Transfer Learning | MobileNetV2 feature extractor | Much better results ✅ |
    | Fine-tuning Top Layers | Full optimization | 🔥 Highest Accuracy |

    With **strong data augmentation**, the model generalizes well even with a limited dataset.

    ---

    ### 🎯 App Features
    ✅ Upload your own flower images  
    ✅ Get instant predictions with confidence %  
    ✅ Sleek UI with sidebar navigation  
    ✅ Works offline once model is downloaded  

    ---

    ### 🌺 Try It Out!
    Go to **Flower Classification** in the sidebar and upload any image of:
    - Daisy
    - Dandelion
    - Rose
    - Sunflower
    - Tulip

    This system will instantly tell you what flower it is! 🌟

    ---

    ### 📡 Future Enhancements
    - Add more flower species
    - Probability charts & visual explanations
    - Live camera support for real-time recognition
    - Deploy to cloud / mobile app

    ---
    """)

    st.info("👈 Use the Navigation Sidebar to explore the app!")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ---
    ## 🌺 Intelligent Flower Recognition Using Deep Learning

    This project is developed as a **complete Machine Learning pipeline** for classifying
    flower species using **Transfer Learning**.

    It consists of multiple major components:
    - ✅ Dataset collection and organization
    - ✅ Data augmentation for better generalization
    - ✅ Custom CNN model training from scratch
    - ✅ Transfer Learning using MobileNetV2
    - ✅ Fine-tuning of top trainable layers
    - ✅ Testing and evaluation metrics
    - ✅ Deployment using Streamlit as an interactive app

    ---
    ### 📚 Dataset Description
    The dataset used is the **TensorFlow Flower Photos Dataset** containing **5 categories**:
    - 🌼 **Daisy**
    - 🌻 **Dandelion**
    - 🌹 **Rose**
    - 🌞 **Sunflower**
    - 🌷 **Tulip**

    Images vary by:
    - Lighting & brightness
    - Background complexity
    - Flower colors & shapes
    - Natural environmental noise

    This makes the classification task challenging and ideal for computer vision research.

    ---
    ### 🧠 Technology Stack
    This system integrates:
    - **TensorFlow + Keras** for modeling
    - **NumPy, Matplotlib, scikit-learn** for analysis
    - **MobileNetV2** pretrained on ImageNet
    - **Python** for the entire workflow
    - **Streamlit** for UI deployment

    The model benefits from **feature reuse** — instead of learning from scratch,
    it uses patterns already learned from millions of images.

    ---
    ### 🔍 Techniques Applied
    | Method | Goal | Outcome |
    |--------|------|---------|
    | Data Augmentation | Increase variation | Higher generalization ✅ |
    | Transfer Learning | Leverage existing CNN | Better accuracy ✅ |
    | Fine-tuning | Optimize highest layers | Peak performance 🔥 |

    The final model is lightweight, fast, and reliable.

    ---
    ### 📈 Results & Performance
    - ✅ Significant improvement over scratch CNN
    - ✅ Achieved strong validation accuracy
    - ✅ Able to classify unseen flower images confidently
    - ✅ Reduced overfitting using augmentation + fine-tuning

    A detailed evaluation (confusion matrix & report) was performed to validate predictions.

    ---
    ### 🎯 Objective & Impact
    This system aims to:
    - Help students and researchers learn real DL workflows
    - Assist botanists in quick species identification
    - Inspire AI-driven agricultural applications
    - Provide an educational tool for learners 🌟

    ---
    ### 🚀 Future Enhancements
    ✅ Add more flower species / datasets  
    ✅ Real-time camera capture  
    ✅ Explainable AI visualization (Grad-CAM)  
    ✅ Mobile app & cloud deployment  

    ---
    ### 👨‍💻 Developer
    **Developed by:** *Ankita Mukherjee (23EC8020)*  
    Passionate about **AI, Computer Vision, and Deep Learning Applications** ❤️  
    """)

# -------------------------------
# Flower Classification Page
# -------------------------------
else:
    st.title("🌺 Flower Image Classification")

    uploaded_file = st.file_uploader(
        "Upload a Flower Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Flower"):
            with st.spinner("Classifying... Please wait"):
                pred_class, confidence = predict_image(img)

            st.success(f"Prediction: 🌸 **{pred_class}**")
            st.write(f"Confidence: **{confidence:.2f}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Ankita Mukherjee (23EC8020)**")
