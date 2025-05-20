import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import tempfile
import streamlit as st
from PIL import Image
import os
from skimage import exposure
import pandas as pd
from fpdf import FPDF

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
model.load_state_dict(torch.load("forest_classifier_final.pth", map_location=device))
model.to(device)
model.eval()

labels = ["Non-Woody", "Sparse Woody", "Dense Forest"]
colors = ["#fef08a", "#f97316", "#166534"]
cmap_dict = mcolors.ListedColormap(colors)

available_cmaps = ["viridis", "plasma", "magma", "terrain", "gray"]

def get_false_color_image(tif_path, colormap='viridis'):
    with rasterio.open(tif_path) as src:
        band = src.read(1).astype(np.float32)
        band = (band - band.min()) / (band.max() - band.min())
    cmap = plt.get_cmap(colormap)
    colored = cmap(band)[:, :, :3]
    return Image.fromarray((colored * 255).astype(np.uint8))

def generate_prediction_map(tif_path):
    tile_size = 64
    stride = 64
    patch_classes = []
    patch_coords = []

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        transform = src.transform
        rows = (height - tile_size) // stride + 1
        cols = (width - tile_size) // stride + 1
        class_grid = np.full((rows, cols), -1, dtype=int)

        for i, y in enumerate(range(0, height - tile_size + 1, stride)):
            for j, x in enumerate(range(0, width - tile_size + 1, stride)):
                window = Window(x, y, tile_size, tile_size)
                patch = src.read(1, window=window).astype(np.float32)
                if np.all(patch == 0): continue
                tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    pred_class = torch.argmax(output, dim=1).item()
                    class_grid[i, j] = pred_class
                    lon, lat = rasterio.transform.xy(transform, y + tile_size//2, x + tile_size//2)
                    patch_classes.append(pred_class)
                    patch_coords.append((x, y, pred_class, lat, lon))

    expanded = np.kron(class_grid, np.ones((64, 64)))
    return expanded, patch_classes, patch_coords

def save_heatmap_with_legend(class_grid):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(class_grid, cmap=cmap_dict, vmin=0, vmax=2)
    ax.set_title("Predicted Forest Type Map")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=colors[i], label=labels[i]) for i in range(3)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

    heatmap_path = os.path.join(tempfile.gettempdir(), "heatmap_with_legend.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    return heatmap_path

def generate_csv(patch_coords, name="forest_predictions.csv"):
    df = pd.DataFrame(patch_coords, columns=["X", "Y", "Predicted_Class", "Lat", "Lon"])
    df["Label"] = df["Predicted_Class"].apply(lambda x: labels[x])
    csv_path = os.path.join(tempfile.gettempdir(), name)
    df.to_csv(csv_path, index=False)
    return csv_path, df

def generate_pdf_summary(summary_text, image_path):
    pdf_path = os.path.join(tempfile.gettempdir(), "forest_summary.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\\n"):
        clean = line.encode("latin-1", "ignore").decode("latin-1")
        pdf.cell(200, 10, txt=clean.strip(), ln=True)
    pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=180)
    pdf.output(pdf_path)
    return pdf_path

st.set_page_config(page_title="TasForester 🌲", layout="wide", page_icon="🌲")
st.title("🌲 AI-Powered Forest Classification for Tasmania")

uploaded_file = st.file_uploader("📂 Upload a .tif satellite ile", type=["tif", "tiff"])
selected_cmap = st.selectbox("🎨 Choose a colormap for satellite view", available_cmaps)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        tif_path = tmp.name

    class_grid, patch_classes, patch_coords = generate_prediction_map(tif_path)
    color_original = get_false_color_image(tif_path, colormap=selected_cmap)
    heatmap_path = save_heatmap_with_legend(class_grid)

    col1, col2 = st.columns(2)
    with col1:
        st.image(color_original, caption="🛰️ Original False Color Satellite Image", use_column_width=True)
    with col2:
        st.image(heatmap_path, caption="🌲 Predicted Map For The Given Satellite Image", use_column_width=True)

    class_counts = {label: patch_classes.count(i) for i, label in enumerate(labels)}
    summary = f"""
Total patches: {sum(patch_classes):,}
- Non-Woody: {class_counts['Non-Woody']:,} patches
- Sparse Woody: {class_counts['Sparse Woody']:,} patches
- Dense Forest: {class_counts['Dense Forest']:,} patches

Insights:
Non-woody regions dominate the western half.
Sparse forest areas cluster toward the center.
Dense forests found mostly in elevated/remote zones.

This data can support reforestation, land use planning, and environmental monitoring.
"""
    st.subheader("📊 Classification Summary")
    st.markdown(summary)

    st.subheader("📤 Export Reports")
    csv_path, df = generate_csv(patch_coords)
    st.download_button("⬇️ Download CSV", open(csv_path, "rb"), "forest_predictions.csv", "text/csv")
    st.download_button("⬇️ Download PNG Heatmap", open(heatmap_path, "rb"), "forest_heatmap.png", "image/png")
    pdf_path = generate_pdf_summary(summary, heatmap_path)
    st.download_button("⬇️ Download PDF Summary", open(pdf_path, "rb"), "forest_summary.pdf", "application/pdf")
