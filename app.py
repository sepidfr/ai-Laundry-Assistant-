"""
AI Laundry Sorter – Multitask ConvNeXt Demo
------------------------------------------
4-Head Model:
  • COLOR        (garments only)
  • FABRIC       (garments only)
  • WASH_CYCLE   (garments only)
  • IS_CLOTHING  (0 = NON_CLOTHING, 1 = CLOTHING)

The model weights are downloaded directly from Google Drive (public share).
"""

import os
import io
import gdown
import datetime
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

import streamlit as st


# ============================================================
# 0) LOAD LABEL FILES
# ============================================================
LABELS_CSV = "wash_labels.csv"
HEADER_IMG = "ai.jpg"
MODEL_LOCAL = "best_model3_wash.pt"       # local filename after download

# Google Drive model file
file_id = "1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Ensure necessary files exist
if not os.path.exists(LABELS_CSV):
    st.error("❌ Missing file: wash_labels.csv — upload it to your project folder.")
    st.stop()

df_all = pd.read_csv(LABELS_CSV)

# Basic checks
required_cols = ["is_clothing","color_label","fabric_label","wash_cycle_label"]
for col in required_cols:
    if col not in df_all.columns:
        st.error(f"❌ Missing required column in CSV: {col}")
        st.stop()


# ============================================================
# 1) MAPPINGS (CLOTHING-ONLY for 3 heads)
# ============================================================
df_cloth = df_all[df_all["is_clothing"] == 1].copy()

num_color = df_cloth["color_label"].nunique()
num_fabric = df_cloth["fabric_label"].nunique()
num_wash = df_cloth["wash_cycle_label"].nunique()
num_iscloth = 2


idx2color = dict(zip(df_cloth["color_label"], df_cloth["color_group"]))
idx2fabric = dict(zip(df_cloth["fabric_label"], df_cloth["fabric_group"]))
idx2wash = dict(zip(df_cloth["wash_cycle_label"], df_cloth["wash_cycle"]))
idx2iscloth = {0: "NON_CLOTHING", 1: "CLOTHING"}


# ============================================================
# 2) IMAGE TRANSFORMS
# ============================================================
IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# 3) MODEL DEFINITION (4-head ConvNeXt)
# ============================================================
BACKBONE = "convnext_tiny"

class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self, num_color, num_fabric, num_wash, num_iscloth):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        self.head_color = nn.Linear(feat_dim, num_color)
        self.head_fabric = nn.Linear(feat_dim, num_fabric)
        self.head_wash = nn.Linear(feat_dim, num_wash)
        self.head_iscloth = nn.Linear(feat_dim, num_iscloth)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash(feat),
            self.head_iscloth(feat),
        )


# ============================================================
# 4) LOAD MODEL (Download if needed)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_LOCAL):
    st.info("⬇ Downloading model from Google Drive…")
    gdown.download(gdrive_url, MODEL_LOCAL, quiet=False)

model = WashMultiTaskConvNeXt(
    num_color=num_color,
    num_fabric=num_fabric,
    num_wash=num_wash,
    num_iscloth=num_iscloth,
).to(device)

state = torch.load(MODEL_LOCAL, map_location=device)
model.load_state_dict(state)
model.eval()


# ============================================================
# 5) PREDICT FUNCTION
# ============================================================
def predict_single(pil_img):
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        lc, lf, lw, lis = model(x)

        pc = lc.argmax(1).item()
        pf = lf.argmax(1).item()
        pw = lw.argmax(1).item()
        pis = lis.argmax(1).item()

        conf_c = F.softmax(lc,1).max().item()
        conf_f = F.softmax(lf,1).max().item()
        conf_w = F.softmax(lw,1).max().item()
        conf_is = F.softmax(lis,1).max().item()

    # Non-garment detection
    if pis == 0:
        return {
            "is_clothing": "NON_CLOTHING",
            "color": "—",
            "fabric": "—",
            "wash": "Not applicable (image is not clothing)",
        }

    return {
        "is_clothing": "CLOTHING",
        "color": idx2color.get(pc, f"color_{pc}"),
        "fabric": idx2fabric.get(pf, f"fabric_{pf}"),
        "wash": idx2wash.get(pw, f"wash_{pw}"),
    }


# ============================================================
# 6) STREAMLIT UI
# ============================================================
st.set_page_config(page_title="AI Laundry Sorter", page_icon="🧺")

if os.path.exists(HEADER_IMG):
    st.image(HEADER_IMG, use_container_width=True)

st.title("AI Laundry Sorter – 4-Head Model")
st.caption("Automatic color, fabric, wash-cycle recommendation + non-clothing detection")

uploaded = st.file_uploader("Upload a clothing image", type=["jpg","jpeg","png"])

if uploaded:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(pil_img, use_container_width=True)

    result = predict_single(pil_img)

    with col2:
        st.subheader("AI Prediction")
        st.markdown(f"**Is Clothing:** {result['is_clothing']}")
        st.markdown(f"**Color Group:** {result['color']}")
        st.markdown(f"**Fabric Group:** {result['fabric']}")
        st.markdown(f"**Wash Program:** {result['wash']}")

    # Log usage
    log_row = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "file_name": uploaded.name,
        "is_clothing": result["is_clothing"],
        "color": result["color"],
        "fabric": result["fabric"],
        "wash": result["wash"],
    }])

    if os.path.exists("demo_log.csv"):
        old = pd.read_csv("demo_log.csv")
        pd.concat([old, log_row]).to_csv("demo_log.csv", index=False)
    else:
        log_row.to_csv("demo_log.csv", index=False)

else:
    st.info("Upload an image to begin.")
