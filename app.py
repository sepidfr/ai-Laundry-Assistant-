"""
AI Laundry Sorter – Streamlit Demo (no gdown)

This app:
  • Downloads best_model3_wash.pt from Google Drive on first run
  • Loads the trained multitask ConvNeXt model (COLOR + FABRIC + WASH_CYCLE)
  • Accepts a single clothing image as input
  • Predicts:
        - Color group   (LIGHT / DARK / COLOR / COLORFUL)
        - Fabric group  (COTTON / LINEN / WOOL / SILK / SYNTHETIC / ...)
        - Washing program (human-readable text)
"""

import os
import io
import datetime

import requests
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

import streamlit as st

# ============================================================
# 0) Core paths and Google Drive model download
# ============================================================

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV  = os.path.join(ROOT_DIR, "wash_labels.csv")
MODEL_PATH  = os.path.join(ROOT_DIR, "best_model3_wash.pt")
HEADER_IMG  = os.path.join(ROOT_DIR, "ai.jpg")

# Your Google Drive model link:
# https://drive.google.com/file/d/1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx/view?usp=drive_link
MODEL_DRIVE_ID = "1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx"


def _get_confirm_token(resp):
    """Extract confirmation token from Google Drive response cookies (if any)."""
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None


def _save_response_content(resp, destination, chunk_size=32768):
    """Stream response content to a local file."""
    with open(destination, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_file_from_google_drive(file_id: str, destination: str):
    """
    Download a file from Google Drive using only 'requests'
    (works also for large files with confirmation token).
    """
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status {response.status_code}")

    _save_response_content(response, destination)


def ensure_model_downloaded():
    """Download model weights from Google Drive if not present locally."""
    if os.path.exists(MODEL_PATH):
        return
    print("Model weights not found locally. Downloading from Google Drive...")
    download_file_from_google_drive(MODEL_DRIVE_ID, MODEL_PATH)
    print("Model download complete:", MODEL_PATH)


# ============================================================
# 1) Load label metadata
# ============================================================

if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"wash_labels.csv not found at {LABELS_CSV}")

df_all = pd.read_csv(LABELS_CSV)

# Use only clothing rows for label mappings
if "is_clothing" in df_all.columns:
    df_cloth = df_all[df_all["is_clothing"] == 1].copy()
else:
    df_cloth = df_all.copy()

for col in ["color_label", "fabric_label", "wash_cycle_label"]:
    df_cloth[col] = df_cloth[col].astype(int)

num_color_classes  = df_cloth["color_label"].nunique()
num_fabric_classes = df_cloth["fabric_label"].nunique()
num_wash_classes   = df_cloth["wash_cycle_label"].nunique()

color_map_df  = df_cloth[["color_label", "color_group"]].drop_duplicates()
fabric_map_df = df_cloth[["fabric_label", "fabric_group"]].drop_duplicates()
wash_map_df   = df_cloth[["wash_cycle_label", "wash_cycle"]].drop_duplicates()

color_map  = dict(zip(color_map_df["color_label"],  color_map_df["color_group"]))
fabric_map = dict(zip(fabric_map_df["fabric_label"], fabric_map_df["fabric_group"]))
wash_map   = dict(zip(wash_map_df["wash_cycle_label"], wash_map_df["wash_cycle"]))

# Optional: richer verbose descriptions for washing programs
wash_full_description = {
    "wool/delicate": "Wool / Delicate – 20–30°C, very gentle cycle, low spin.",
    "delicate/hand-wash": "Delicate / Hand-wash – 20°C, gentle agitation, low spin.",
    "normal": "Normal – 30–40°C, everyday cottons and basics.",
    "normal/delicate": "Normal / Delicate – 30°C, slightly reduced agitation.",
    "synthetic/easy-care": "Synthetic / Easy-care – 30°C, anti-crease profile.",
}

# ============================================================
# 2) Model + preprocessing
# ============================================================

IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

BACKBONE_NAME = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    """
    ConvNeXt backbone with three heads:
      • COLOR
      • FABRIC
      • WASH_CYCLE
    """
    def __init__(self, num_color, num_fabric, num_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head_color      = nn.Linear(feat_dim, num_color)
        self.head_fabric     = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)

    def forward(self, x):
        feat = self.backbone(x)
        logits_color      = self.head_color(feat)
        logits_fabric     = self.head_fabric(feat)
        logits_wash_cycle = self.head_wash_cycle(feat)
        return logits_color, logits_fabric, logits_wash_cycle


# Ensure model weights exist (download if needed)
ensure_model_downloaded()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ============================================================
# 3) Prediction helper
# ============================================================

def predict_single_image(pil_img: Image.Image):
    """
    Run multitask model on a single image and return
    color / fabric / wash predictions (with simple confidence check).
    """
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)

        probs_c = F.softmax(logits_c, dim=1)
        probs_f = F.softmax(logits_f, dim=1)
        probs_w = F.softmax(logits_w, dim=1)

        max_pc, idx_c = probs_c.max(dim=1)
        max_pf, idx_f = probs_f.max(dim=1)
        max_pw, idx_w = probs_w.max(dim=1)

        max_pc = float(max_pc.item())
        max_pf = float(max_pf.item())
        max_pw = float(max_pw.item())

        LOW_CONF = 0.50
        low_conf = min(max_pc, max_pf, max_pw) < LOW_CONF

    pc = int(idx_c.item())
    pf = int(idx_f.item())
    pw = int(idx_w.item())

    color_name  = color_map.get(pc,  f"Unknown (id={pc})")
    fabric_name = fabric_map.get(pf, f"Unknown (id={pf})")

    wash_key = wash_map.get(pw, f"wash_{pw}")
    wash_text = wash_full_description.get(wash_key, wash_key)

    if low_conf:
        wash_text = "[Low confidence] " + wash_text

    return {
        "color":  color_name,
        "fabric": fabric_name,
        "wash":   wash_text,
    }

# ============================================================
# 4) Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption("Multitask ConvNeXt model for automatic color, fabric, and wash-program recommendations.")

    uploaded_file = st.file_uploader(
        "Upload a clothing image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(pil_img, use_container_width=True)

        result = predict_single_image(pil_img)

        with col2:
            st.subheader("AI Recommendation")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # Simple usage log in app directory
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        log_path = os.path.join(ROOT_DIR, "demo_usage_log.csv")
        log_row = pd.DataFrame([{
            "timestamp": ts,
            "image_name": uploaded_file.name,
            "pred_color": result["color"],
            "pred_fabric": result["fabric"],
            "pred_wash": result["wash"],
        }])

        if os.path.exists(log_path):
            old = pd.read_csv(log_path)
            pd.concat([old, log_row], ignore_index=True).to_csv(log_path, index=False)
        else:
            log_row.to_csv(log_path, index=False)

        st.success("Prediction logged.")
    else:
        st.info("Upload a garment image to receive an automatic washing recommendation.")


if __name__ == "__main__":
    main()
