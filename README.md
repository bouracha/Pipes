# Project: Diagram Processing Pipeline

## Overview
This project automates the processing of diagram PDFs using YOLOv5 and OCR. The pipeline extracts individual diagrams from a PDF, runs object detection, builds a graphical network, and verifies design rules using an AI assistant.

## Requirements
### **1. Install Conda (if not installed)**
This project uses Conda for environment management. Install Conda from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

### **2. Create and Activate the Conda Environment**
```bash
conda create -n pipes python=3.10 -y
conda activate pipes
```

### **3. Install Dependencies**
```bash
pip install torch torchvision torchaudio  # PyTorch
pip install opencv-python  # OpenCV
pip install ultralytics  # YOLOv5
pip install pytesseract  # OCR
pip install python-docx  # Word document processing
pip install networkx pandas numpy matplotlib  # Graph processing
```

## **File and Directory Setup**
1. **Place `sop.docx` in the main project directory.**
2. **Place your diagram PDF in the `diagrams/` directory.**

## **Pipeline Execution**
### **Step 1: Convert PDF to Images**
Run the following command to extract diagrams from the PDF:
```bash
python3 pdf_to_images.py
```
This will generate multiple `.jpg` files in the `diagrams/` folder.

### **Step 2: Process Each Diagram**
For each extracted image (e.g., `diagram_1.jpg`), run:
```bash
python3 main.py diagram_1.jpg
```
Repeat this for each image (e.g., `diagram_2.jpg`, `diagram_3.jpg`, etc.).

## **Outputs**
The pipeline produces the following results:
- **Graphical Network Output:**
  - A file called `networkx_graph.csv` is generated in:
    ```
    yolov5/runs/detect/exp{run_number}/networkx_graph.csv
    ```
  - This contains the graphical connections and specifications extracted via OCR.

- **Design Rule Verification Response:**
  - The AI assistant (Claude) provides a response determining whether the design follows the specified rules based on OCR-extracted information.

### **Example Inference (Diagram_1)**
```text
# Component Specification Comparison Report

I'll extract specifications from the SOP and search for matching components in the diagram_ocr dataset.

## F-715 A and B Particulate Filters

**Found in diagram_ocr as "F-~7iS A & B PARTICULATE FILTER SEPARATOR"**

Specifications from SOP:
- Pressure: 275 psig
- Temperature: 100°F

Specifications from diagram_ocr:
- Operating: 230 PSIG
- Design: 275 PSIG @ 100F
- MDMT: -20F @ 275 PSIG
- Design Cap: 180 GPM @ 2.25 PSID CLEAN

**Result: INCORRECT**
- Discrepancy: The diagram shows operating pressure of 230 PSIG, while the SOP only lists the design pressure of 275 PSIG.
- The design values match (275 PSIG, 100°F), but there's an additional operating value in the diagram.

## Other Components

The following components from SOP were not found in the diagram_ocr dataset:
- V-745 Stabilizer Tower
- E-742 Exchanger (Shell and Tube)
- AC-746 After Cooler

As per instructions, I'm not reporting on these missing components.

## Explicit Design Rule Violations Found

No explicit design rule violations were found. The only discrepancy noted is that the diagram shows an operating pressure (230 PSIG) for the F-715 A & B Particulate Filters, which is below the design pressure (275 PSIG) listed in both the SOP and diagram. This is not a violation, as equipment should operate below its design pressure.
```

## **Notes**
- Ensure all dependencies are installed before running the scripts.
- The script dynamically selects the latest detection folder (`exp{run_number}`).
- If any issues arise, verify that `pytesseract` and `tesseract-ocr` are installed correctly (refer to [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)).

## **Example Workflow**
```bash
conda activate pipes
python3 pdf_to_images.py
python3 main.py diagram_1.jpg
python3 main.py diagram_2.jpg
python3 main.py diagram_3.jpg
```
This will extract diagrams from the PDF, process each diagram, build the graphical network, and validate design rules.

## **Contact & Support**
For any issues or improvements, feel free to raise an issue in the project repository or contact the maintainers.

