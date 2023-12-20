import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model("model_wildfire.h5")

# Function to preprocess input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(448, 448))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def main():
    st.title("Wildfire Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and make predictions
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(prediction[0])
        class_names = ["Wildfire","Non-Wildfire"]
        result = class_names[predicted_class]

        # Display the result
        st.write(f"Prediction: {result}")

        # Display the confidence scores for each class
        st.write("Confidence Scores:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i]:.2%}")

if __name__ == "__main__":
    main()
