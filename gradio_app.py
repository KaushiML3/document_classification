import gradio as gr
from pathlib import Path
from PIL import Image

# Import your model classes (adjust import paths as needed)
from app.src.vit_load import VITDocumentClassifier
from app.src.vgg16_load import VGGDocumentClassifier
from app.src.constant import vit_model_path, vit_mlb_path, vgg_model_path, vgg_mlb_path

# Load models once at startup
vit_model = VITDocumentClassifier(vit_model_path, vit_mlb_path)
vgg_model = VGGDocumentClassifier(vgg_model_path, vgg_mlb_path)

def predict_vit(image, cut_off):
    if image is None:
        return "Please upload an image."
    temp_path = "temp_vit_image.png"
    image.save(temp_path)
    result = vit_model.predict(Path(temp_path), cut_off)
    return f"ViT Prediction: {result}"

def predict_vgg(image):
    if image is None:
        return "Please upload an image."
    temp_path = "temp_vgg_image.png"
    image.save(temp_path)
    result = vgg_model.predict(Path(temp_path))
    return f"VGG16 Prediction: {result}"

with gr.Blocks() as demo:
    gr.Markdown("# Document Classification Demo\nUpload an image and choose a model to classify it.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            cut_off = gr.Slider(0, 1, value=0.5, label="ViT Cutoff Threshold")
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", interactive=False)
    with gr.Row():
        vit_btn = gr.Button("Predict with ViT Model")
        vgg_btn = gr.Button("Predict with VGG16 Model")

    vit_btn.click(fn=predict_vit, inputs=[image_input, cut_off], outputs=result_output)
    vgg_btn.click(fn=predict_vgg, inputs=image_input, outputs=result_output)

if __name__ == "__main__":
    demo.launch() 