import requests
import json
import os
import pandas as pd
import argparse

def load_csv_data(csv_path):
    """Reads a CSV file and converts it to a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def load_text_data(text_path):
    """Reads a text file and returns its content as a string."""
    try:
        with open(text_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {text_path}: {e}")
        return None

def format_csv_data_for_claude(df, title):
    """Formats CSV data into a readable text format for Claude."""
    if df is None or df.empty:
        return f"{title}: No data available.\n"

    formatted_text = f"### {title} ###\n"
    formatted_text += df.to_string(index=False)  # Convert DataFrame to a clean text format
    return formatted_text + "\n"

def send_to_claude(diagram_ocr, sop):
    """Sends formatted CSV data and text data to Claude for analysis."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is missing. Set it as an environment variable.")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    message_content = (
        "I have two datasets from object detection on a Piping & Instrumentation Diagram (P&ID).\n"
        "Your task is to:\n"
        "1. **Start by looking at the SOP dataset**. For each component listed in SOP, extract all its specifications (e.g., pressure, temperature, flow rate, material, etc.).\n"
        "2. **Then, search for each SOP component in the `diagram_ocr` dataset** to check if the same component is present.\n"
        "   - Match component names approximately (e.g., 'E-742 Exchanger (Shell)' vs. 'E-742').\n"
        "   - If a component from SOP is found in `diagram_ocr`, extract all its specifications from `diagram_ocr`.\n"
        "3. Compare the values of all specifications:\n"
        "   - If all specifications match, mark it as correct.\n"
        "   - If any specification does not match, mark it as incorrect and specify the discrepancy.\n"
        "   - **Missing components in `diagram_ocr` should not be reported**—only check those that exist in both datasets.\n"
        "4. Return a structured report listing only:\n"
        "   - The components from SOP that were found in `diagram_ocr`.\n"
        "   - Whether their specifications were correct or incorrect.\n"
        "   - If incorrect, list the specific mismatches with expected vs. found values.\n\n"
        "The SOP table is **not formatted correctly**, so do NOT rely on typical table parsing.\n"
        "Instead, treat the SOP table as **a list where the first column is equipment name, and subsequent columns contain various specifications**.\n"
        "**Ignore 'NaN' values and missing columns**—they are just formatting artifacts.\n"
        "Ensure you **extract values based on pattern recognition, NOT table formatting**."
        "The end of your reply should say if there was any explicit design rule violations found."
    )

    message_content += format_csv_data_for_claude(diagram_ocr, "diagram_ocr") + "\n"
    message_content += f"### SOP ###\n{sop}\n "

    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process diagram OCR and SOP data for Claude.")
    parser.add_argument("diagram_prefix", help="Prefix of the diagram file (e.g., 'diagram_3' for 'diagram_3_ocr.csv')")
    args = parser.parse_args()

    # Construct filenames dynamically
    csv_path = f"{args.diagram_prefix}_ocr.csv"
    txt_path = "extracted_text.txt"

    diagram_ocr = load_csv_data(csv_path)
    sop = load_text_data(txt_path)

    result = send_to_claude(diagram_ocr, sop)

    if result:
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                print(f"Claude says: {item.get('text')}")

        print("\nFull response:")
        print(json.dumps(result, indent=2))
