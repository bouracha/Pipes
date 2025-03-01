# File: sliding_detect.py
# Place this file in the root directory of the repository

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add YOLOv5 to path
sys.path.insert(0, 'yolov5')

# Import directly from the main detect module to avoid version conflicts
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device


def load_model(weights_path, device=''):
    """Load YOLOv5 model from weights file"""
    device = select_device(device)
    # Changed: removed map_location argument based on the error
    model = attempt_load(weights_path)
    model = model.to(device)
    return model, device


def process_window(model, window, device, conf_thres, iou_thres, window_offset=(0, 0)):
    # Ensure consistent dimensions by explicitly resizing to model's expected input size
    img_size = model.stride.max() * 32  # Default YOLOv5 image size

    # Force resize to desired dimensions
    img = letterbox(window, img_size, stride=model.stride, auto=model.pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # Normalize

    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections and adjust coordinates based on window offset
    results = []
    for det in pred:  # per image
        if len(det):
            # Adjust coordinates based on window_offset
            adjusted_det = det.clone()
            adjusted_det[:, 0] += window_offset[0]  # x1
            adjusted_det[:, 2] += window_offset[0]  # x2
            adjusted_det[:, 1] += window_offset[1]  # y1
            adjusted_det[:, 3] += window_offset[1]  # y2
            results.append(adjusted_det)

    return results


def nms(boxes, scores, iou_threshold):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_indices = []
    while sorted_indices.size > 0:
        # Pick the box with highest score
        current_index = sorted_indices[0]
        keep_indices.append(current_index)

        # Get IoU of current box with remaining boxes
        ious = calculate_iou(boxes[current_index], boxes[sorted_indices[1:]])

        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return keep_indices


def calculate_iou(box, boxes):
    """Calculate IoU between a box and multiple boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Intersection area
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Box areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # IoU
    iou = intersection_area / (box_area + boxes_area - intersection_area)

    return iou


def sliding_window_detection(image, model, device, window_size=(640, 640), overlap=0.2, conf_thres=0.25,
                             iou_thres=0.45):
    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate stride (non-overlapping part of each window)
    stride_h = int(window_size[0] * (1 - overlap))
    stride_w = int(window_size[1] * (1 - overlap))

    all_detections = []

    # Iterate over all positions
    for y in range(0, h - window_size[0] + stride_h, stride_h):
        # Adjust y to not exceed image bounds
        y = min(y, h - window_size[0])

        for x in range(0, w - window_size[1] + stride_w, stride_w):
            # Adjust x to not exceed image bounds
            x = min(x, w - window_size[1])

            # Extract window
            window = image[y:y + window_size[0], x:x + window_size[1]]

            # Process window
            window_detections = process_window(model, window, device, conf_thres, iou_thres, window_offset=(x, y))

            # Add to all detections
            all_detections.extend(window_detections)

    return all_detections


def save_results_to_csv(detections, image_path, output_dir, class_names):
    """Save detection results to CSV file"""
    filename = Path(image_path).stem
    csv_path = Path(output_dir) / f"{filename}.csv"

    results = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        results.append({
            'image_name': f"{filename}.png",
            'class': class_name,
            'x_min': int(x1),
            'y_min': int(y1),
            'x_max': int(x2),
            'y_max': int(y2),
            'confidence': float(conf)
        })

    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print(f"No detections found for {filename}")


def process_directory(source_dir, weights_path, output_dir, window_size=600, overlap=100, conf_thres=0.25,
                      iou_thres=0.45):
    """Process all images in a directory using sliding window detection"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, device = load_model(weights_path)

    # Get class names
    class_names = model.names if hasattr(model, 'names') else model.module.names if hasattr(model, 'module') else {}

    # Get all image files
    image_files = []
    source_path = Path(source_dir)
    if source_path.is_dir():
        image_files = [p for p in source_path.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        # Handle wildcard patterns like "diagrams/*"
        import glob
        image_files = [Path(p) for p in glob.glob(source_dir) if Path(p).suffix.lower() in ['.jpg', '.jpeg', '.png']]

    if not image_files:
        print(f"No image files found in {source_dir}")
        return

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"Processing: {img_path}")

        # Load the image first
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue

        # Now call with correct parameter order
        detections = sliding_window_detection(
            img,  # Actual image (not path)
            model,  # YOLOv5 model
            device,
            window_size=window_size,
            overlap=overlap,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        # Save results
        save_results_to_csv(detections, img_path, output_dir, class_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='diagrams', help='source directory')
    parser.add_argument('--output', type=str, default='results/csv', help='output directory')
    parser.add_argument('--window-size', type=int, default=600, help='sliding window size')
    parser.add_argument('--overlap', type=int, default=100, help='overlap between windows')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    args = parser.parse_args()

    process_directory(
        args.source,
        args.weights,
        args.output,
        window_size=args.window_size,
        overlap=args.overlap,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )