import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

# Parse command-line argument for base directory
parser = argparse.ArgumentParser(description="Process images and detect lines.")
parser.add_argument("base_dir", type=str, help="Base directory for the detection results (e.g., yolov5/runs/detect/exp17/)")
args = parser.parse_args()

# Use the base directory from the command-line argument
base_dir = args.base_dir

# Update file paths to use the base directory
csv_path = f"{base_dir}/reconstructed_predictions.csv"
output_csv_path = f"{base_dir}/detected_lines.csv"

df = pd.read_csv(csv_path)

# Load the image
image_path = "diagrams/diagram_2.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Error: Unable to load image {image_path}. Check file path.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a mask to exclude detected components
mask = np.ones_like(gray) * 255  # White background

# Black out component bounding boxes
for _, row in df[df["Image Name"] == "diagram_1.jpg"].iterrows():
    x_min, y_min, x_max, y_max = int(row["Xmin"]), int(row["Ymin"]), int(row["Xmax"]), int(row["Ymax"])
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 0, -1)  # Black out detected components

# Apply the mask before edge detection
filtered_gray = cv2.bitwise_and(gray, gray, mask=mask)
edges = cv2.Canny(filtered_gray, 100, 200)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength= 5, maxLineGap=50)

# Filter lines to only keep horizontal and vertical ones
filtered_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx < 1 or dy < 1:  # Accept only horizontal (small dy) or vertical (small dx) lines
            filtered_lines.append((x1, y1, x2, y2))

# Draw component bounding boxes (Green)
for (x_min, y_min, x_max, y_max) in zip(df["Xmin"], df["Ymin"], df["Xmax"], df["Ymax"]):
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Fixed

# Draw only filtered horizontal and vertical lines (Blue)
for (x1, y1, x2, y2) in filtered_lines:
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


lines_df = pd.DataFrame(filtered_lines, columns=["X1", "Y1", "X2", "Y2"])
lines_df.to_csv(output_csv_path, index=False)
print(f"Saved {len(filtered_lines)} detected lines to {output_csv_path}")

# Show final image with masked components and filtered lines
#plt.figure(figsize=(10, 6))
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.title("Filtered Horizontal & Vertical Lines (After Masking Components)")
#plt.axis("off")
#plt.show()
