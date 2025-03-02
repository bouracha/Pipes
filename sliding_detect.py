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
from yolov5.utils.general import non_max_suppression



def letterbox(img, new_shape=(640, 640), stride=300, auto=True, scale_fill=False, scale_up=True):
    """
    Resize image while maintaining aspect ratio, adding padding if necessary.

    Parameters:
        img (np.array): Input image
        new_shape (tuple): Target image size (width, height)
        stride (int): Stride for resizing
        auto (bool): Whether to automatically pad to the stride size
        scale_fill (bool): Stretch image to fill new_shape
        scale_up (bool): Allow image to be scaled up

    Returns:
        img (np.array): Resized image with padding
        ratio (tuple): (width ratio, height ratio)
        (dw, dh): (width padding, height padding)
    """
    shape = img.shape[:2]  # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute the scaling ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scale_up:
        r = min(r, 1.0)  # Prevent scaling up

    # Compute new unpadded width and height
    new_unpad = (int(round(float(shape[1] * r))), int(round(float(shape[0] * r))))


    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # Width, Height padding
    if auto:  # Minimize padding by distributing evenly
        dw, dh = dw % stride, dh % stride
    dw /= 2  # Divide padding equally on both sides
    dh /= 2

    # Resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border padding
    top, bottom = int(round(float(dh) - 0.1)), int(round(float(dh) + 0.1))
    left, right = int(round(float(dw) - 0.1)), int(round(float(dw) + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, (r, r), (dw, dh)


def load_model(weights_path, device=''):
    """Load YOLOv5 model from weights file"""
    device = select_device(device)
    # Changed: removed map_location argument based on the error
    model = attempt_load(weights_path)
    model = model.to(device)
    return model, device


def process_window(model, window, device, conf_thres, iou_thres, window_offset=(0, 0), window_id=0):
    img_size = model.stride.max() * 32  # Default YOLOv5 image size

    # Use the new letterbox function
    img, _, _ = letterbox(window, new_shape=(img_size, img_size), stride=model.stride.max(), auto=True)

    # Convert image format
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize

    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    print(f"DEBUG: NMS output for window {window_id}:")
    for i, det in enumerate(pred):
        if len(det):
            print(f"  Detection {i}: {det.cpu().numpy()}")

    results = []
    for det in pred:
        if len(det):
            det[:, 0] += window_offset[0]  # x1
            det[:, 2] += window_offset[0]  # x2
            det[:, 1] += window_offset[1]  # y1
            det[:, 3] += window_offset[1]  # y2

            det_list = det.cpu().numpy().tolist()
            results.append(det_list)

            # Draw bounding boxes on the window
            for x1, y1, x2, y2, conf, cls_id in det_list:
                cv2.rectangle(window, (int(x1 - window_offset[0]), int(y1 - window_offset[1])),
                              (int(x2 - window_offset[0]), int(y2 - window_offset[1])), (0, 255, 0), 2)
                cv2.putText(window, f"Class {int(cls_id)}: {conf:.2f}",
                            (int(x1 - window_offset[0]), int(y1 - window_offset[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed window with detections
    os.makedirs("debug_inferences", exist_ok=True)
    window_filename = f"debug_inferences/window_{window_id}.jpg"
    cv2.imwrite(window_filename, window)
    print(f"DEBUG: Saved inference image at {window_filename}")

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


def sliding_window_detection(image, model, device, window_size=(600, 600), overlap=0.5, conf_thres=0.25,
                             iou_thres=0.45):
    h, w = image.shape[:2]

    # Ensure overlap is correctly interpreted
    if overlap >= 1:
        overlap /= 100.0  # Convert from percentage to fraction

    # Compute stride for 50% overlap
    stride_h = int(window_size[0] * (1 - overlap))  # Should be 50% of window size
    stride_w = int(window_size[1] * (1 - overlap))

    print(
        f"DEBUG: Image size (h={h}, w={w}), Window size={window_size}, Stride (h={stride_h}, w={stride_w}), Overlap={overlap}")

    all_detections = []

    os.makedirs("debug_windows", exist_ok=True)
    window_id = 0  # Counter for naming saved images

    for y in range(0, h - window_size[0] + stride_h, stride_h):
        y = min(y, h - window_size[0])

        for x in range(0, w - window_size[1] + stride_w, stride_w):
            x = min(x, w - window_size[1])
            print(f"DEBUG: Processing column start x={x}, window_id={window_id}")

            # Extract window
            window = image[y:y + window_size[0], x:x + window_size[1]]

            # Save raw window for debugging
            raw_window_filename = f"debug_windows/window_{window_id}.jpg"
            cv2.imwrite(raw_window_filename, window)
            print(f"DEBUG: Saved raw window {window_id} at {raw_window_filename}")

            # Process window
            window_detections = process_window(model, window, device, conf_thres, iou_thres, window_offset=(x, y),
                                               window_id=window_id)

            # Print detections for each window
            print(f"DEBUG: Detections for window {window_id}:")
            for i, detection_list in enumerate(window_detections):
                for det in detection_list:
                    print(f"  Detection {i}: {det}")  # Print each detection

            all_detections.extend(window_detections)
            window_id += 1  # Increment counter

    return all_detections


def save_results_to_csv(detections, image_path, output_dir, class_names):
    """Save detection results to CSV file"""
    filename = Path(image_path).stem
    csv_path = Path(output_dir) / f"{filename}.csv"

    results = []
    for detection_list in detections:  # <-- Iterate through list of detections
        for det in detection_list:  # <-- Now iterate through each detection
            print(f"DEBUG: Individual detection = {det}, length = {len(det)}")  # Debugging

            if len(det) < 6:
                print(f"WARNING: Skipping malformed detection: {det}")
                continue  # Skip incorrect detections

            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            class_name = class_names.get(cls_id, f"class_{cls_id}")

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
        print(f"No valid detections found for {filename}")


def process_directory(source_dir, weights_path, output_dir, window_size=(600, 600), overlap=300, conf_thres=0.25,
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
    parser.add_argument('--window-size', type=tuple, default=(600,600), help='sliding window size')
    parser.add_argument('--overlap', type=int, default=100, help='overlap between windows')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    args = parser.parse_args()

    process_directory(
        args.source,
        args.weights,
        args.output,
        window_size=args.window_size,
        overlap=args.overlap / 100.0 if args.overlap > 1 else args.overlap,  # <-- Convert percentage to fraction
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
