import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import csv
import argparse
import os

def load_data(csv_path_1, csv_path_2, ocr_path):
    """Load bounding box, line, and OCR text data from CSV files."""
    components = pd.read_csv(csv_path_1)
    lines = pd.read_csv(csv_path_2)
    if os.path.exists(ocr_path):
        ocr_texts = pd.read_csv(ocr_path)

        # Rename columns to standard format
        ocr_texts.rename(columns={"X Position": "x", "Y Position": "y", "Text": "text"}, inplace=True)
    else:
        ocr_texts = pd.DataFrame(columns=["x", "y", "text"])  # Match expected names

    return components, lines, ocr_texts

def is_point_in_bbox(x, y, bbox):
    """Check if a point (x, y) is inside a bounding box."""
    return bbox['Xmin'] <= x <= bbox['Xmax'] and bbox['Ymin'] <= y <= bbox['Ymax']

def find_bbox(x, y, components):
    """Find the bounding box containing the given point (x, y)."""
    for _, bbox in components.iterrows():
        if is_point_in_bbox(x, y, bbox):
            return f"({bbox['Xmin']}, {bbox['Ymin']}, {bbox['Xmax']}, {bbox['Ymax']})"
    return None

def assign_text_to_nodes(G, ocr_texts):
    """Assign OCR text to the nearest node based on proximity."""
    for _, row in ocr_texts.iterrows():
        text_x, text_y, text = row["x"], row["y"], row["text"]
        closest_node = None
        min_distance = float("inf")

        for node, pos in nx.get_node_attributes(G, 'pos').items():
            node_x, node_y = pos
            distance = ((node_x - text_x) ** 2 + (node_y - text_y) ** 2) ** 0.5  # Euclidean distance

            if distance < min_distance and distance < 500:
                min_distance = distance
                closest_node = node

        if closest_node:
            G.nodes[closest_node]["text"] = text  # Assign text to the nearest node

def build_networkx_graph(components, lines, ocr_texts):
    """Build a NetworkX graph with all nodes, including OCR annotations."""
    G = nx.Graph()

    # Add nodes (ensure all bounding boxes are included, even if isolated)
    for _, bbox in components.iterrows():
        node_key = f"({bbox['Xmin']}, {bbox['Ymin']}, {bbox['Xmax']}, {bbox['Ymax']})"
        x_center = (bbox['Xmin'] + bbox['Xmax']) / 2
        y_center = (bbox['Ymin'] + bbox['Ymax']) / 2
        G.add_node(node_key, pos=(x_center, y_center))  # Store node position

    # Add edges (connections between bounding boxes)
    for _, line in lines.iterrows():
        start_bbox = find_bbox(line['X1'], line['Y1'], components)
        end_bbox = find_bbox(line['X2'], line['Y2'], components)

        if start_bbox and end_bbox and start_bbox != end_bbox:
            G.add_edge(start_bbox, end_bbox)

    # Assign OCR text to the nearest nodes
    assign_text_to_nodes(G, ocr_texts)

    return G

def save_networkx_graph(G, output_path="networkx_graph.csv"):
    """Save the NetworkX graph to a CSV file, including isolated nodes and OCR text."""
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Bounding Box (Node)", "X Center", "Y Center", "Connected Nodes", "Text"])

        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            text = G.nodes[node].get("text", "")  # Retrieve text if available
            neighbors = ", ".join(G.neighbors(node)) if list(G.neighbors(node)) else ""
            writer.writerow([node, x, y, neighbors, text])

    print(f"NetworkX graph saved to {output_path}")

def plot_graph(G):
    """Plot the graph with node positions and attached OCR text."""
    plt.figure(figsize=(10, 8))

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=200, edge_color="black", alpha=0.7)

    # Draw node labels near the nodes
    for node, (x, y) in pos.items():
        text = G.nodes[node].get("text", "")
        plt.text(x, y, node, fontsize=6, ha="right", va="bottom", color="blue")
        if text:
            plt.text(x, y - 10, text, fontsize=8, ha="center", va="top", color="red")  # OCR text in red

    plt.title("Graph Representation with OCR Annotations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()

def main():
    """Main function to run the script with command line input."""
    parser = argparse.ArgumentParser(description="Generate a NetworkX graph with OCR text annotations.")
    parser.add_argument("base_dir", help="Base directory containing the CSV files (e.g., yolov5/runs/detect/exp17/)")
    parser.add_argument("diagram_name", help="Diagram name prefix (e.g., 'diagram_3') for OCR file lookup")

    args = parser.parse_args()
    base_dir = args.base_dir
    diagram_name = args.diagram_name

    # Construct full paths
    csv_path_1 = os.path.join(base_dir, "reconstructed_predictions.csv")
    csv_path_2 = os.path.join(base_dir, "detected_lines.csv")
    ocr_path = os.path.join(f"{diagram_name}_ocr.csv")
    networkx_graph_path = os.path.join(base_dir, "networkx_graph.csv")

    # Ensure required files exist
    if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
        print(f"Error: One or both CSV files not found in {base_dir}")
        return

    components, lines, ocr_texts = load_data(csv_path_1, csv_path_2, ocr_path)
    G = build_networkx_graph(components, lines, ocr_texts)

    # Save graph and plot
    save_networkx_graph(G, networkx_graph_path)
    plot_graph(G)

if __name__ == "__main__":
    main()
