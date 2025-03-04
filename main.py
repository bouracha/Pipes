import os
import sys
import glob
import subprocess
import shutil

def get_latest_exp_number(detect_folder):
    """Get the highest-numbered experiment folder in yolov5/runs/detect/"""
    exp_folders = glob.glob(os.path.join(detect_folder, 'exp*'))
    exp_numbers = [int(folder.split('exp')[-1]) for folder in exp_folders if folder.split('exp')[-1].isdigit()]
    return max(exp_numbers) if exp_numbers else 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <image_filename>")
        sys.exit(1)

    image_filename = sys.argv[1]
    output_sections = "output_sections/"
    detect_folder = "yolov5/runs/detect/"

    # Step 1: Split the image
    subprocess.run(
        ["python3", "split_and_stitch.py", "split", f"diagrams/{image_filename}", output_sections, "--overlap=0.0",
         "--window_size", "700"], check=True)

    # Step 2: Run YOLOv5 detection
    subprocess.run(
        ["python3", "yolov5/detect.py", "--weights", "yolov5/best.pt", "--source", "output_sections/*", "--save-csv"],
        check=True)

    # Get the latest experiment folder number
    latest_exp = get_latest_exp_number(detect_folder)
    if latest_exp == 0:
        latest_exp = ''
    latest_exp_folder = os.path.join(detect_folder, f"exp{latest_exp}")

    # Copy split_metadata.csv to the latest experiment folder
    metadata_file = os.path.join(output_sections, "split_metadata.csv")
    if os.path.exists(metadata_file):
        shutil.copy(metadata_file, latest_exp_folder)

    # Step 3: Stitch the sections back together
    subprocess.run(
        ["python3", "split_and_stitch.py", "stitch", latest_exp_folder, "reconstructed.jpg", "--window_size", "700"],
        check=True)

    # Step 4: Run connections.py
    subprocess.run(["python3", "connections.py", latest_exp_folder], check=True)

    # Step 5: Run build_graph.py
    subprocess.run(["python3", "build_graph.py", latest_exp_folder, image_filename], check=True)

    # Step 6: Run OCR_diagram.py
    subprocess.run(["python3", "OCR_diagram.py", image_filename], check=True)

    # Step 6: Run OCR_sop.py
    subprocess.run(["python3", "OCR_sop.py"], check=True)

    # Step 7: Use Anthropic to check design rule violations based on extracted information
    subprocess.run(["python3", "cross_reference.py", image_filename[:-4]], check=True)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()