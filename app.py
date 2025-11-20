
"""
AI Laundry Sorter – Streamlit Demo (4-Head Model, Google Drive weights)

This app:
  • Loads the trained multitask ConvNeXt model (COLOR + FABRIC + WASH_CYCLE + IS_CLOTHING)
  • Accepts a single clothing image as input
  • Predicts:
        - Color group   (LIGHT / DARK / COLOR / COLORFUL)
        - Fabric group  (COTTON / LINEN / WOOL / SILK / SYNTHETIC)
        - Washing program (human-readable text, no numeric labels in UI)
  • Uses the IS_CLOTHING head + confidence heuristics to reject clearly non-garment images

Model weights are downloaded once from Google Drive using the public share link.
"""

import os
import io
import datetime
import requests

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

import streamlit as st

# ============================================================
# 0) Paths & Google Drive config
# ============================================================
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))

LABELS_CSV = os.path.join(ROOT_DIR, "wash_labels.csv")
HEADER_IMG = os.path.join(ROOT_DIR, "ai.jpg")

# محلی که مدل دانلودشده را ذخیره می‌کنیم
CKPT_PATH  = os.path.join(ROOT_DIR, "best_model_wash.pt")

# فایل لاگ استفاده از دمو
DEMO_LOG   = os.path.join(ROOT_DIR, "demo_usage_log.csv")

# Google Drive file ID (از share-link تو گرفته شده)
# Link: https://drive.google.com/file/d/1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx/view?usp=drive_link
GDRIVE_FILE_ID = "1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx"

assert os.path.exists(LABELS_CSV), f"Labels CSV not found: {LABELS_CSV}"


# ============================================================
# 0.1) Helper: download model weights from Google Drive if needed
# ============================================================
def download_file_from_google_drive(file_id: str, destination: str):
    """
    Minimal Google Drive downloader using `requests`.
    Assumes the file is shared as 'Anyone with link'.
    """
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    with requests.get(url, params=params, stream=True) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ensure_checkpoint_present():
    """
    If best_model_wash.pt is not present locally, download it from Google Drive.
    """
    if os.path.exists(CKPT_PATH):
        return
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    print("Downloading model weights from Google Drive...")
    download_file_from_google_drive(GDRIVE_FILE_ID, CKPT_PATH)
    print("Model downloaded to:", CKPT_PATH)


# ============================================================
# 1) Label metadata (for mapping indices → names)
# ============================================================
df_all = pd.read_csv(LABELS_CSV)
assert "is_clothing" in df_all.columns, "Column 'is_clothing' missing in wash_labels.csv"

# ---------------- Clothing subset for COLOR / FABRIC heads -------------
df_clothing = df_all[df_all["is_clothing"] == 1].copy()

df_clothing["color_label"]  = df_clothing["color_label"].astype(int)
df_clothing["fabric_label"] = df_clothing["fabric_label"].astype(int)

num_color_classes  = df_clothing["color_label"].nunique()
num_fabric_classes = df_clothing["fabric_label"].nunique()

# ---------------- WASH_CYCLE head: use ALL rows (۵ کلاس) ---------------
wash_all = df_all[["wash_cycle_label", "wash_cycle"]].dropna(subset=["wash_cycle_label", "wash_cycle"]).copy()
wash_all["wash_cycle_label"] = wash_all["wash_cycle_label"].astype(int)

num_wash_classes    = wash_all["wash_cycle_label"].nunique()  # باید 5 باشد
num_iscloth_classes = 2  # {0: NON_CLOTHING, 1: CLOTHING}

print("num_color_classes :", num_color_classes)
print("num_fabric_classes:", num_fabric_classes)
print("num_wash_classes  :", num_wash_classes)

# Clean mappings: label → name
color_map_df  = df_clothing[["color_label",  "color_group"]].drop_duplicates()
fabric_map_df = df_clothing[["fabric_label", "fabric_group"]].drop_duplicates()
wash_map_df   = wash_all[["wash_cycle_label", "wash_cycle"]].drop_duplicates()

color_map  = dict(zip(color_map_df["color_label"],  color_map_df["color_group"]))
fabric_map = dict(zip(fabric_map_df["fabric_label"], fabric_map_df["fabric_group"]))
wash_map   = dict(zip(wash_map_df["wash_cycle_label"], wash_map_df["wash_cycle"]))

# Optional: richer verbal descriptions for washing programs
# Make sure keys here match your actual 'wash_cycle' strings.
wash_full_description = {
    "wool/delicate":       "Wool / Delicate – ~20°C, ultra-gentle agitation, very low spin.",
    "delicate/hand-wash":  "Delicate / Hand-wash – 20–30°C, gentle cycle, low spin, ideal for silk and fine fabrics.",
    "normal":              "Normal – 30–40°C, standard agitation and spin for everyday cotton/synthetics.",
    "normal/delicate":     "Normal / Delicate – 30°C, slightly gentler mechanical action, medium spin.",
    "synthetic/easy-care": "Synthetic / Easy-care – 30°C, anti-crease profile, moderate spin.",
    # اگر اسم‌های دیگری در CSV داری اینجا اضافه/اصلاح کن
}

# ============================================================
# 2) Model + preprocessing transforms
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
    Multitask ConvNeXt backbone (4 heads):
      • head_color      → color_group classification
      • head_fabric     → fabric_group classification
      • head_wash_cycle → wash_cycle classification (۵ کلاس)
      • head_is_cloth   → is_clothing (0 = NON_CLOTHING, 1 = CLOTHING)
    """
    def __init__(self, num_color, num_fabric, num_wash, num_iscloth=2):
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
        self.head_is_cloth   = nn.Linear(feat_dim, num_iscloth)

    def forward(self, x):
        feat = self.backbone(x)
        logits_color      = self.head_color(feat)
        logits_fabric     = self.head_fabric(feat)
        logits_wash_cycle = self.head_wash_cycle(feat)
        logits_is_cloth   = self.head_is_cloth(feat)
        return logits_color, logits_fabric, logits_wash_cycle, logits_is_cloth


# روی استریم‌لیت کلود عملاً روی CPU اجرا می‌شود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مطمئن شو فایل مدل لوکال است؛ اگر نیست از گوگل‌درایو بگیر
ensure_checkpoint_present()

model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
    num_iscloth=num_iscloth_classes,
).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ============================================================
# 3) Single-image prediction helper
# ============================================================
def predict_single_image(pil_img: Image.Image):
    """
    Run the trained multitask model on a single PIL image.

    Returns
    -------
    dict:
      {
        "color":  <predicted color_group or message>,
        "fabric": <predicted fabric_group or message>,
        "wash":   <human-readable wash program description>,
      }
    """
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w, logits_is = model(x)

        probs_c  = F.softmax(logits_c, dim=1)
        probs_f  = F.softmax(logits_f, dim=1)
        probs_w  = F.softmax(logits_w, dim=1)
        probs_is = F.softmax(logits_is, dim=1)  # shape: [1, 2]

        max_pc, idx_c = probs_c.max(dim=1)
        max_pf, idx_f = probs_f.max(dim=1)
        max_pw, idx_w = probs_w.max(dim=1)

        max_pc = max_pc.item()
        max_pf = max_pf.item()
        max_pw = max_pw.item()

        # Probability that image contains clothing (class 1)
        p_is_cloth = probs_is[0, 1].item()

        VERY_LOW_CONF = 0.30
        LOW_CONF      = 0.55

        # Non-garment gate
        if (p_is_cloth < 0.5) and (max_pc < VERY_LOW_CONF) and (max_pf < VERY_LOW_CONF):
            return {
                "color":  "No garment detected",
                "fabric": "No garment detected",
                "wash":   "No washing program suggested — the image does not appear to contain clothing.",
            }

        low_conf_flag = (min(max_pc, max_pf, max_pw) < LOW_CONF) or (p_is_cloth < 0.6)

    # Decode labels
    pc = idx_c.item()
    pf = idx_f.item()
    pw = idx_w.item()

    color_name  = color_map.get(pc,  f"Unknown (id={pc})")
    fabric_name = fabric_map.get(pf, f"Unknown (id={pf})")

    wash_key = wash_map.get(pw, f"wash_{pw}")
    full_wash_text = wash_full_description.get(wash_key, wash_key)

    if low_conf_flag:
        full_wash_text = "[Low confidence] " + full_wash_text

    return {
        "color":  color_name,
        "fabric": fabric_name,
        "wash":   full_wash_text,
    }

# ============================================================
# 4) Streamlit application
# ============================================================
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    # Optional header image
    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption(
        "Multitask ConvNeXt (4-head) for automatic color, fabric, "
        "and washing program recommendations."
    )

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

        # Log usage
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        log_row = pd.DataFrame([{
            "timestamp":   ts,
            "image_name":  uploaded_file.name,
            "pred_color":  result["color"],
            "pred_fabric": result["fabric"],
            "pred_wash":   result["wash"],
        }])

        if os.path.exists(DEMO_LOG):
            old = pd.read_csv(DEMO_LOG)
            pd.concat([old, log_row], ignore_index=True).to_csv(DEMO_LOG, index=False)
        else:
            log_row.to_csv(DEMO_LOG, index=False)

        st.success("Prediction logged.")
    else:
        st.info("Upload a garment image to receive an automatic washing recommendation.")

if __name__ == "__main__":
    main()
