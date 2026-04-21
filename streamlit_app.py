import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

def detect_fractures(image):
    """
    Send image to API and return detection results.
    """
    if image is None:
        return "Please upload an image first.", None

    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    try:
        # Send POST request to API
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(PREDICT_ENDPOINT, files=files)
        response.raise_for_status()  # Raise exception for bad status codes

        result = response.json()

        # Format results
        width, height = result['image_size']['width'], result['image_size']['height']
        result_text = f"Image size: {width}x{height}\n"
        result_text += f"Confidence threshold: {result['confidence_threshold']}\n\n"

        if result['num_detections'] == 0:
            result_text += "No fractures detected."
            return result_text, image

        result_text += f"Found {result['num_detections']} detections:\n\n"

        detections = []
        for detection in result['detections']:
            bbox = detection['bbox']
            detections.append({
                "label": detection['label'],
                "confidence": detection['confidence'],
                "bbox": [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]
            })

            result_text += f"• {detection['label']}: {detection['confidence']:.1%} confidence\n"
            result_text += f"  Position: ({bbox['x_min']:.0f}, {bbox['y_min']:.0f}) to ({bbox['x_max']:.0f}, {bbox['y_max']:.0f})\n\n"

        # Draw bounding boxes on image
        annotated_image = draw_boxes_on_image(image, detections)

        return result_text, annotated_image

    except requests.exceptions.RequestException as e:
        return f"Error connecting to API: {str(e)}. Make sure the FastAPI server is running.", None
    except Exception as e:
        return f"Error processing response: {str(e)}", None

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

def main():
    st.title("🦴 Bone Fracture Detection")
    st.write("Upload an X-ray image to detect bone fractures using AI. The model will identify different types of fractures and their locations.")
    st.write(f"**API Endpoint:** {PREDICT_ENDPOINT}")

    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detection button
        if st.button("Detect Fractures"):
            with st.spinner("Analyzing image... (calling API)"):
                result_text, annotated_image = detect_fractures(image)

            # Display results
            st.subheader("Detection Results")
            st.text_area("Results", result_text, height=200)

            if annotated_image is not None:
                st.subheader("Annotated Image")
                st.image(annotated_image, caption="Detected Fractures", use_column_width=True)

if __name__ == "__main__":
    main()