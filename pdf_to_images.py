from pdf2image import convert_from_path
import os

# Define paths
pdf_path = "diagrams/diagram.pdf"
output_dir = "diagrams/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path, dpi=300)

# Save each page as a separate JPG file
for i, image in enumerate(images):
    image_path = os.path.join(output_dir, f"diagram_{i+1}.jpg")
    image.save(image_path, "JPEG")

print(f"Saved {len(images)} images in {output_dir}")
