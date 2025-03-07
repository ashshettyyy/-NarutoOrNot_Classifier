import gradio as gr
import os
import platform
from pathlib import Path

# Fix for PosixPath issue on Windows
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Import fastai after the path fix
from fastai.learner import load_learner

# Create placeholder prediction function that doesn't load model
def predict_image_simple(img):
    # Return dummy results without loading model
    return {
        "Naruto": 0.8,
        "Not Naruto": 0.2
    }

# Create Gradio interface with the simple function
interface = gr.Interface(
    fn=predict_image_simple,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Naruto Classifier (Test)",
    description="Upload an image (model not loaded in this test)"
)

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    interface.launch(server_name="0.0.0.0", server_port=7860)
    print("Interface should be running now")