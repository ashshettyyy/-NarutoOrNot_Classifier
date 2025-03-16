import os
import gradio as gr
import torch
import numpy as np
import fastai
import platform
from fastai.learner import load_learner
import pickle
import sys
from pathlib import Path

# At the beginning of your script
print("Script starting...")  

if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Import fastai after the path fix
from fastai.learner import load_learner

# Function to load the model safely
def load_model():
    try:
        # Use the correct path to your model file
        model_path = Path('export.pkl')
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            # Alternative loading method
            with open('export.pkl', 'rb') as f:
                learn = pickle.load(f)
            return learn
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            sys.exit(1)

# Function to make predictions
def predict_image(img):
    # Load the model
    learn = load_model()
    
    # Get prediction
    pred, pred_idx, probs = learn.predict(img)
    
    # Return results
    return {
        "Naruto": float(probs[0]),  # Adjust indices based on your classes
        "Not Naruto": float(probs[1])
    }

# Right before launching the interface
print("About to launch Gradio interface...")

# Create Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Naruto Classifier",
    description="Upload an image to check if it's from Naruto or not"
)

# Launch the app
print("Launching interface on http://127.0.0.1:7860")
if __name__ == "__main__":
    interface.launch(share=True)
    print("Interface should be running now")