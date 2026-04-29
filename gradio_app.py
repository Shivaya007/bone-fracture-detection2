import gradio as gr
import torch
import numpy as np
from PIL import Image
import io
from torchvision.ops import nms
from transformers import DetrImageProcessor, DetrForObjectDetection
import os

# Configuration
MODEL_PATH = "custom-model"
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.3
PROCESSOR_SHORTEST_EDGE = 1000
PROCESSOR_LONGEST_EDGE = 1333
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
print(f"Loading model from '{MODEL_PATH}' on {DEVICE} ...")
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Build id → label map from the model config
id2label = {int(k): v for k, v in model.config.id2label.items()}
print(f"Model loaded. Classes: {id2label}")

def detect_fractures(image):
    """
    Process the uploaded image and return detection results.
    """
    if image is None:
        return "Please upload an image first.", None

    # Convert PIL image to RGB if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    width, height = image.size

    # Run inference
    with torch.no_grad():
        inputs = image_processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": PROCESSOR_SHORTEST_EDGE, "longest_edge": PROCESSOR_LONGEST_EDGE},
        ).to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([[height, width]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        if len(results["scores"]) > 0:
            keep = nms(results["boxes"], results["scores"], NMS_IOU_THRESHOLD)
            results = {
                "scores": results["scores"][keep],
                "labels": results["labels"][keep],
                "boxes": results["boxes"][keep],
            }

    # Format results
    detections = []
    result_text = f"Image size: {width}x{height}\n"
    result_text += f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n\n"

    if len(results["scores"]) == 0:
        result_text += "No fractures detected."
        return result_text, image

    result_text += f"Found {len(results['scores'])} detections:\n\n"

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x_min, y_min, x_max, y_max = box.tolist()
        label_name = id2label.get(label.item(), "unknown")
        confidence = score.item()

        detections.append({
            "label": label_name,
            "confidence": confidence,
            "bbox": [x_min, y_min, x_max, y_max]
        })

        result_text += f"• {label_name}: {confidence:.1%} confidence\n"
        result_text += f"  Position: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})\n\n"

    # Draw bounding boxes on image
    annotated_image = draw_boxes_on_image(image, detections)

    return result_text, annotated_image

def draw_boxes_on_image(image, detections):
    """
    Draw bounding boxes on the image using PIL.
    """
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']

        # Draw rectangle
        color = colors[i % len(colors)]
        draw.rectangle(bbox, outline=color, width=3)

        # Draw label
        text = f"{label}: {confidence:.1%}"
        # Get text size (approximate)
        text_bbox = draw.textbbox((bbox[0], bbox[1]-25), text)
        draw.rectangle([bbox[0], bbox[1]-25, text_bbox[2], text_bbox[3]], fill=color)
        draw.text((bbox[0], bbox[1]-25), text, fill='white')

    return image

# Create Gradio interface
demo = gr.Interface(
    fn=detect_fractures,
    inputs=gr.Image(label="Upload X-ray Image", type="pil"),
    outputs=[
        gr.Textbox(label="Detection Results", lines=15),
        gr.Image(label="Annotated Image")
    ],
    title="🦴 Bone Fracture Detection",
    description="Upload an X-ray image to detect bone fractures using AI. The model will identify different types of fractures and their locations.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    print("Model loaded, launching server...")
    print("Gradio server started on http://0.0.0.0:7863")
    demo.launch(server_name="0.0.0.0", server_port=7865, theme=gr.themes.Soft(), share=False, inbrowser=False)