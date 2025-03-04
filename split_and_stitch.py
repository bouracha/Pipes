import os
import cv2
import csv
import numpy as np


def split(image_path, output_folder, window_size=600, overlap=0.5):
    """
    Splits an image into overlapping sections and saves them along with a CSV file
    containing metadata for reassembly.

    :param image_path: Path to the input JPG image.
    :param output_folder: Directory where split images and CSV will be saved.
    :param window_size: Size of each square section (default 600x600 pixels).
    :param overlap: Overlap percentage between sections (default 50%).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Ensure the path is correct and it's a valid JPG file.")

    img_height, img_width, _ = image.shape
    step_size = int(window_size * (1 - overlap))  # Calculate step size

    csv_data = []
    section_id = 0

    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            # Ensure the section is within bounds
            x_end = min(x + window_size, img_width)
            y_end = min(y + window_size, img_height)
            section = np.zeros((window_size, window_size, 3), dtype=np.uint8)
            section[:y_end - y, :x_end - x] = image[y:y_end, x:x_end]

            section_filename = f"section_{section_id}.jpg"
            section_path = os.path.join(output_folder, section_filename)
            cv2.imwrite(section_path, section)
            csv_data.append([section_filename, x, y, x_end - x, y_end - y])
            section_id += 1

    # Save CSV metadata
    csv_path = os.path.join(output_folder, "split_metadata.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "X", "Y", "Width", "Height"])  # Header
        writer.writerows(csv_data)

    print(f"Splitting complete. Sections saved in {output_folder} with metadata in split_metadata.csv")


def stitch(input_folder, output_image_path, window_size=600):
    """
    Stitches image sections back together based on metadata and transforms bbox coordinates.

    :param input_folder: Directory containing split images and metadata CSV.
    :param output_image_path: Path to save the reconstructed image.
    :param window_size: Size of each square section (default 600x600 pixels).
    """
    csv_path = os.path.join(input_folder, "split_metadata.csv")
    predictions_csv_path = os.path.join(input_folder, "predictions.csv")
    output_predictions_csv = os.path.join(input_folder, "reconstructed_predictions.csv")

    if not os.path.exists(csv_path):
        raise ValueError("Metadata CSV not found in input folder.")

    # Read metadata
    sections = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 5:
                continue  # Skip invalid rows
            sections.append((row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])))

    if not sections:
        raise ValueError("No valid sections found in metadata CSV.")

    # Determine output image size
    max_x = max(x + w for _, x, _, w, _ in sections)
    max_y = max(y + h for _, _, y, _, h in sections)

    # Create empty canvas
    stitched_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # Stitch sections together
    for filename, x, y, w, h in sections:
        section_path = os.path.join(input_folder, filename)
        section = cv2.imread(section_path)
        if section is None:
            print(f"Warning: Skipping missing section {filename}")
            continue
        stitched_image[y:y + h, x:x + w] = section[:h, :w]

    # Save the stitched image
    cv2.imwrite(output_image_path, stitched_image)
    print(f"Stitching complete. Image saved at {output_image_path}")

    # Transform bbox coordinates if predictions.csv exists
    if os.path.exists(predictions_csv_path):
        transformed_bboxes = []
        with open(predictions_csv_path, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                if len(row) < 7:
                    continue  # Skip invalid rows
                filename, label, confidence, xmin, ymin, xmax, ymax = row
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

                # Find corresponding section offset
                for section_filename, x_offset, y_offset, _, _ in sections:
                    if section_filename == filename:
                        global_xmin = xmin + x_offset
                        global_ymin = ymin + y_offset
                        global_xmax = xmax + x_offset
                        global_ymax = ymax + y_offset
                        transformed_bboxes.append(
                            [filename, label, confidence, global_xmin, global_ymin, global_xmax, global_ymax])
                        break

        # Save transformed bbox coordinates
        with open(output_predictions_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Name", "Label", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax"])  # Header
            writer.writerows(transformed_bboxes)
        print(f"Transformed bbox coordinates saved to {output_predictions_csv}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and Stitch an image.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Split command
    split_parser = subparsers.add_parser("split", help="Split an image into sections.")
    split_parser.add_argument("image_path", type=str, help="Path to the input JPG image.")
    split_parser.add_argument("output_folder", type=str, help="Folder to save split images and metadata.")
    split_parser.add_argument("--window_size", type=int, default=600, help="Size of each section (default 600).")
    split_parser.add_argument("--overlap", type=float, default=0.5, help="Overlap percentage (default 50%).")

    # Stitch command
    stitch_parser = subparsers.add_parser("stitch", help="Stitch sections back together.")
    stitch_parser.add_argument("input_folder", type=str, help="Folder containing split images and metadata.")
    stitch_parser.add_argument("output_image_path", type=str, help="Path to save the reconstructed image.")
    stitch_parser.add_argument("--window_size", type=int, default=600, help="Size of each section (default 600).")

    args = parser.parse_args()

    if args.command == "split":
        split(args.image_path, args.output_folder, args.window_size, args.overlap)
    elif args.command == "stitch":
        stitch(args.input_folder, args.output_image_path, args.window_size)
