import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

# Ensure Tesseract is installed and set the path if necessary
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path as needed


def preprocess_image(image_path):
    """Load and preprocess image for better OCR accuracy."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found or unreadable.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image, gray, thresh


def extract_text_and_positions(image):
    """Extract text and bounding box positions from the image using Tesseract."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel format for Tesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    extracted_text = []
    text_positions = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # Ignore empty text results
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            extracted_text.append(text)
            text_positions.append((x + w // 2, y + h // 2))  # Use center of bounding box

    return extracted_text, np.array(text_positions)


def save_text_to_csv(extracted_text, text_positions, output_csv):
    """Save extracted text and positions to a CSV file."""
    df = pd.DataFrame({
        'Text': extracted_text,
        'X Position': text_positions[:, 0],
        'Y Position': text_positions[:, 1]
    })
    df.to_csv(output_csv, index=False)
    print(f"Extracted text and positions saved to {output_csv}")


def plot_text_on_image(image, extracted_text, text_positions):
    """Overlay extracted text onto the image at their detected positions."""
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for text, (x, y) in zip(extracted_text, text_positions):
        plt.text(x, y, text, fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis("off")
    plt.show()


"""Main function to run the script with command line input."""
parser = argparse.ArgumentParser(description="OCR.")
parser.add_argument("diagram", help="Name of diagram")

args = parser.parse_args()
diagram = args.diagram

# Construct full paths
image_path = os.path.join("diagrams/", diagram)
output_csv = f"{os.path.splitext(diagram)[0]}_ocr.csv"

# Example usage
original_img, gray_img, preprocessed_img = preprocess_image(image_path)
extracted_text, text_positions = extract_text_and_positions(gray_img)

save_text_to_csv(extracted_text, text_positions, output_csv)

# Plot extracted text
#plot_text_on_image(original_img, extracted_text, text_positions)