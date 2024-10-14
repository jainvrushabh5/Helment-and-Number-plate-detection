import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the YOLO models
model_all = YOLO('best.pt')  # Model for helmet, license plate, and motorcyclist
model_np = YOLO('best_1.pt')  # Model for number plate detection

def predict(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Perform inference with both models
    results_all = model_all(img_array)
    results_np = model_np(img_array)

    # Merge detections into a single image
    # Plot results from the first model (all detections)
    combined_image = None
    for r in results_all:
        im_array = r.plot()
        combined_image = im_array

    # Plot results from the second model (number plate detection) on the same image
    for r in results_np:
        np_array = r.plot()
        if combined_image is not None:
            combined_image = np.maximum(combined_image, np_array)  # Combine both images (max pixel values)

    im = Image.fromarray(combined_image[..., ::-1])  # Convert back to RGB PIL image

    # Initialize counters and confidence lists
    class_counts_all = {i: 0 for i in range(len(model_all.names))}
    class_confidences_all = {i: [] for i in range(len(model_all.names))}
    np_count = 0
    np_confidences = []

    # Process results from the first model (all detections)
    for box in results_all[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_counts_all[cls] += 1
        class_confidences_all[cls].append(conf)

    # Process results from the second model (number plate detection)
    for box in results_np[0].boxes:
        np_count += 1
        np_confidences.append(float(box.conf[0]))

    # Combine the counts from both models into one output string
    output = "Detection Results:\n"
    for i in range(len(model_all.names)):
        count = class_counts_all[i]
        avg_conf = np.mean(class_confidences_all[i]) if class_confidences_all[i] else 0
        if count > 0:  # Only print classes with detections
            output += f"{model_all.names[i]}: {count} detections (Avg. Confidence: {avg_conf:.2f})\n"

    # Add number plate detection results from the second model
    if np_count > 0:  # Only print number plate detection if there are any
        avg_np_conf = np.mean(np_confidences) if np_confidences else 0
        output += f"Number Plates: {np_count} detections (Avg. Confidence: {avg_np_conf:.2f})\n"

    return im, output

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Detected Image"),
        gr.Textbox(label="Detection Results")
    ],
    title="Helmet, License Plate, and Motorcyclist Detection",
    description="Upload an image to detect helmets, license plates, and motorcyclists using two specialized models."
)

# Launch the interface
iface.launch(share=True)
