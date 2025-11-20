"""
AI Laundry Sorter – Streamlit Demo (4-Head Model)

This app:
  • Loads a trained 4-head ConvNeXt model
      - COLOR
      - FABRIC
      - WASH_CYCLE
      - IS_CLOTHING
  • Accepts a single clothing image as input
  • Predicts:
        - Color group   (LIGHT / DARK / COLOR / COLORFUL)
        - Fabric group  (COTTON / LINEN / WOOL / SILK / SYNTHETIC, ...)
        - Washing program (human-readable text)
  • Uses IS_CLOTHING + confidence thresholds to reject clearly non-garment images.
"""

import os
import io
import datetime

import pandas as pd
from PIL import Image
import streamlit as st

# ============================================================
# 0) Try to import heavy DL packages (torch, torchvision, timm)
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    import timm
except ModuleNotFoundError as e:
    st.error(
        f"Missing Python package: `{e.name}`.\n\n"
        "Please make sure it is listed in `requirements.txt` in your GitHub "
        "repository (for example: `torch`, `torchvision`, `timm`) and redeploy the app."
    )
    st.stop()

# ============================================================
# 1) Paths and Google Drive model download
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS_CSV = os.path.join(BASE_DIR, "wash_labels.csv")
HEADER_IMG = os.path.join(BASE_DIR, "ai.jpg")              # optional
DEMO_LOG   = os.path.join(BASE_DIR, "demo_usage_log.csv")

CKPT_PATH  = os.path.join(BASE_DIR, "best_model3_wash.pt")

# Google Drive ID for the model:
# https://drive.google.com/file/d/1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx/view
GDRIVE_FILE_ID = "1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx"


def ensure_model_downloaded():
    """Download model checkpoint from Google Drive using gdown if needed."""
    if os.path.exists(CKPT_PATH):
        return

    try:
        import gdown
    except ModuleNotFoundError:
        st.error(
            "The `gdown` package is not installed.\n\n"
            "Please add `gdown` to `requirements.txt` in your GitHub repository."
        )
        st.stop()

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    st.info("Downloading model weights from Google Drive... (this may take a moment)")
    gdown.download(url, CKPT_PATH, quiet=False)

    if not os.path.exists(CKPT_PATH):
        st.error("Model download failed – checkpoint file not found after download.")
        st.stop()


# Ensure labels CSV exists
if not os.path.exists(LABELS_CSV):
    st.error(
        f"`wash_labels.csv` not found at:\n{LABELS_CSV}\n\n"
        "Please place `wash_labels.csv` in the same folder as `app.py` in your GitHub repo."
    )
    st.stop()

# Make sure model checkpoint is present (download if needed)
ensure_model_downloaded()

# ============================================================
# 2) Label metadata (mapping indices → names)
# ============================================================
df_all = pd.read_csv(LABELS_CSV)

if "is_clothing" not in df_all.columns:
    st.error("Column `is_clothing` is missing in `wash_labels.csv`.")
    st.stop()

# ---------- Clothing subset for COLOR / FABRIC heads ----------
df_clothing = df_all[df_all["is_clothing"] == 1].copy()
if df_clothing.empty:
    st.error("No clothing rows found (`is_clothing == 1`) in `wash_labels.csv`.")
    st.stop()

# Drop rows with missing labels to avoid IntCastingNaNError
df_clothing = df_clothing.dropna(subset=["color_label", "fabric_label"])
if df_clothing.empty:
    st.error(
        "After dropping rows with missing `color_label`/`fabric_label`, "
        "no clothing rows remain in `wash_labels.csv`."
    )
    st.stop()

df_clothing["color_label"]  = df_clothing["color_label"].astype(int)
df_clothing["fabric_label"] = df_clothing["fabric_label"].astype(int)

num_color_classes  = df_clothing["color_label"].nunique()
num_fabric_classes = df_clothing["fabric_label"].nunique()

# ---------- WASH_CYCLE head ----------
wash_all = df_all[["wash_cycle_label", "wash_cycle"]].dropna(
    subset=["wash_cycle_label", "wash_cycle"]
).copy()
wash_all["wash_cycle_label"] = wash_all["wash_cycle_label"].astype(int)

# checkpoint was trained with 5 wash-cycle classes
num_wash_classes    = 5
num_iscloth_classes = 2  # {0: NON_CLOTHING, 1: CLOTHING}

# mappings
color_map_df  = df_clothing[["color_label",  "color_group"]].dropna(subset=["color_group"]).drop_duplicates()
fabric_map_df = df_clothing[["fabric_label", "fabric_group"]].dropna(subset=["fabric_group"]).drop_duplicates()
wash_map_df   = wash_all[["wash_cycle_label", "wash_cycle"]].drop_duplicates()

color_map  = dict(zip(color_map_df["color_label"],  color_map_df["color_group"]))
fabric_map = dict(zip(fabric_map_df["fabric_label"], fabric_map_df["fabric_group"]))
wash_map   = dict(zip(wash_map_df["wash_cycle_label"], wash_map_df["wash_cycle"]))

wash_full_description = {
    "wool/delicate":       "Wool / Delicate – ~20°C, ultra-gentle agitation, very low spin.",
    "delicate/hand-wash":  "Delicate / Hand-wash – 20–30°C, gentle cycle, low spin, ideal for silk and fine fabrics.",
    "normal":              "Normal – 30–40°C, standard agitation and spin for everyday cotton/synthetics.",
    "normal/delicate":     "Normal / Delicate – 30°C, slightly gentler mechanical action, medium spin.",
    "synthetic/easy-care": "Synthetic / Easy-care – 30°C, anti-crease profile, moderate spin.",
}

# ============================================================
# 3) Model + preprocessing transforms
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
      • head_wash_cycle → wash_cycle classification (5 classes)
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


device = torch.device("cpu")

model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
    num_iscloth=num_iscloth_classes,
).to(device)

try:
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    st.error(
        f"Error loading model checkpoint from `{CKPT_PATH}`:\n\n{e}\n\n"
        "Please make sure the checkpoint is compatible with the model architecture."
    )
    st.stop()

model.eval()

# ============================================================
# 4) Single-image prediction helper
# ============================================================
def predict_single_image(pil_img: Image.Image):
    """
    Run the trained multitask model on a single PIL image.
    """
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w, logits_is = model(x)

        probs_c  = F.softmax(logits_c, dim=1)
        probs_f  = F.softmax(logits_f, dim=1)
        probs_w  = F.softmax(logits_w, dim=1)
        probs_is = F.softmax(logits_is, dim=1)  # [1, 2]

        max_pc, idx_c = probs_c.max(dim=1)
        max_pf, idx_f = probs_f.max(dim=1)
        max_pw, idx_w = probs_w.max(dim=1)

        max_pc = max_pc.item()
        max_pf = max_pf.item()
        max_pw = max_pw.item()

        p_is_cloth = probs_is[0, 1].item()

        # ---------- IS_CLOTHING gate (softer: 0.5) ----------
        # If the model is less than 50% sure this is clothing,
        # we treat it as non-garment for the demo.
        if p_is_cloth < 0.50:
            return {
                "color":  "No garment detected",
                "fabric": "No garment detected",
                "wash":   "No washing program suggested — "
                          "the image does not appear to contain clothing.",
            }

        # Low-confidence flag for *garment* predictions
        LOW_CONF = 0.55
        low_conf_flag = (min(max_pc, max_pf, max_pw) < LOW_CONF) or (p_is_cloth < 0.7)

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
# 5) Streamlit application
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
    st.caption(
        "Multitask ConvNeXt (4-head) for automatic color, fabric, "
        "and washing-program recommendations."
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

        ts = datetime.datetime.now().isoformat(timespec="seconds")
        log_row = pd.DataFrame([{
            "timestamp":   ts,
            "image_name":  uploaded_file.name,
            "pred_color":  result["color"],
            "pred_fabric": result["fabric"],
            "pred_wash":   result["wash"],
        }])

        try:
            if os.path.exists(DEMO_LOG):
                old = pd.read_csv(DEMO_LOG)
                pd.concat([old, log_row], ignore_index=True).to_csv(DEMO_LOG, index=False)
            else:
                log_row.to_csv(DEMO_LOG, index=False)
        except Exception:
            pass

        st.success("Prediction done.")
    else:
        st.info("Upload a garment image to receive an automatic washing recommendation.")


if __name__ == "__main__":
    main()
