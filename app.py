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
  • Uses IS_CLOTHING + confidence thresholds + a smart rule-based gate
    to reject clearly non-garment images.
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

# Local files expected in the repo
LABELS_CSV = os.path.join(BASE_DIR, "wash_labels.csv")
HEADER_IMG = os.path.join(BASE_DIR, "ai.jpg")              # optional
DEMO_LOG   = os.path.join(BASE_DIR, "demo_usage_log.csv")

# Model checkpoint will be saved with this name locally
CKPT_PATH  = os.path.join(BASE_DIR, "best_model3_wash.pt")

# Google Drive file ID for the model
# Link: https://drive.google.com/file/d/1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx/view?usp=drive_link
GDRIVE_FILE_ID = "1TxEEeU-uTVS-SYq4uzZISDr4gj7CFUsx"


def ensure_model_downloaded():
    """
    Download the model checkpoint from Google Drive using gdown
    if it does not yet exist in the app folder.

    Important:
    - The Drive file must be shared as:
      'Anyone with the link' -> Viewer.
    """
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

    st.info("Downloading model weights from Google Drive... (this may take a moment)")

    try:
        # Use direct id=... instead of raw URL
        gdown.download(id=GDRIVE_FILE_ID, output=CKPT_PATH, quiet=False)
    except Exception as e:
        st.error(
            "Failed to download the model file from Google Drive.\n\n"
            "Please check that:\n"
            "  • The file is NOT in Trash\n"
            "  • Sharing is set to 'Anyone with the link'\n"
            "  • The File ID is correct: "
            f"{GDRIVE_FILE_ID}\n\n"
            f"Raw error from gdown:\n{e}"
        )
        st.stop()

    if not os.path.exists(CKPT_PATH):
        st.error(
            "Model download finished but `best_model3_wash.pt` was not found.\n"
            "Please verify the Google Drive file and try again."
        )
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

# ----- Class-prior correction for FABRIC (empirical priors from CSV) -----
import numpy as np
fabric_counts = df_clothing["fabric_label"].value_counts().sort_index()
_eps = 1e-6
_full_counts = np.full((num_fabric_classes,), _eps, dtype=np.float32)
for k, v in fabric_counts.items():
    if 0 <= k < num_fabric_classes:
        _full_counts[k] = float(v)
fabric_priors = _full_counts / _full_counts.sum()
fabric_log_priors = torch.from_numpy(np.log(fabric_priors)).float()
PRIOR_ALPHA = 0.8  # strength of prior correction (0..1)

# ----- Find label id for WOOL (case-insensitive contains "wool") -----
wool_label_id = None
for k, v in fabric_map.items():
    if isinstance(v, str) and ("wool" in v.lower()):
        wool_label_id = int(k)
        break

# ----- Find label id for SILK and SYNTHETIC (case-insensitive contains) -----
silk_label_id = None
synthetic_label_id = None
for k, v in fabric_map.items():
    if isinstance(v, str):
        name = v.lower()
        if ("silk" in name) and (silk_label_id is None):
            silk_label_id = int(k)
        # match "synthetic" or common variants like "poly", "polyester"
        if (("synthetic" in name) or ("poly" in name)) and (synthetic_label_id is None):
            synthetic_label_id = int(k)

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
# 4) Single-image prediction helper (TTA + prior correction + fabric rules)
# ============================================================
def _tta_batch_from_pil(pil_img: Image.Image):
    # Build a small TTA set: original, hflip, and a slightly resized+center-crop
    imgs = []
    base = pil_img.copy()
    imgs.append(demo_transform(base))
    imgs.append(demo_transform(base.transpose(Image.FLIP_LEFT_RIGHT)))
    # slightly different scale → more texture robustness
    t_resize = transforms.Resize(int(IMG_SIZE * 1.10))
    t_center = transforms.CenterCrop(IMG_SIZE)
    tta_alt = transforms.Compose([
        t_resize, t_center,
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    imgs.append(tta_alt(base))
    return torch.stack(imgs, dim=0).to(device)  # [T, 3, H, W]


# -------- Robust HSV statistics over garment region (background-invariant) --------
def compute_hsv_stats_masked(pil_img):
    """
    Robust HSV over garment region:
      - Exclude near-white background (RGB>240) OR (S<0.08 & V>0.92)
      - Use median instead of mean
      - Fallback: center crop if mask too small
    Returns (H_med, S_med, V_med, frac_light)
    """
    img = pil_img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    R, G, B = arr[...,0], arr[...,1], arr[...,2]

    # RGB→HSV (vectorized)
    maxc = arr.max(axis=-1); minc = arr.min(axis=-1); delta = maxc - minc + 1e-12
    V = maxc
    S = (delta / (maxc + 1e-12))
    H = np.zeros_like(V)
    mask_r = (maxc == R)
    mask_g = (maxc == G)
    mask_b = (maxc == B)
    H[mask_r] = (G[mask_r]-B[mask_r]) / delta[mask_r]
    H[mask_g] = 2.0 + (B[mask_g]-R[mask_g]) / delta[mask_g]
    H[mask_b] = 4.0 + (R[mask_b]-G[mask_b]) / delta[mask_b]
    H = (H/6.0) % 1.0

    # Background suppression
    near_white_rgb = (R>0.94) & (G>0.94) & (B>0.94)
    near_white_hsv = (S<0.08) & (V>0.92)
    fg = ~(near_white_rgb | near_white_hsv)

    # Fallback if too small
    if fg.sum() < 500:
        h, w = V.shape
        y0, y1 = int(0.1*h), int(0.9*h)
        x0, x1 = int(0.1*w), int(0.9*w)
        fg = np.zeros_like(V, dtype=bool)
        fg[y0:y1, x0:x1] = True

    Hf = H[fg]; Sf = S[fg]; Vf = V[fg]

    # Fraction of pixels that are "very light" (pastel-like)
    pastel_A = (Sf < 0.18) & (Vf > 0.70)
    pastel_B = (Sf < 0.30) & (Vf > 0.88)
    frac_light = float((pastel_A | pastel_B).mean()) if Sf.size else 0.0

    H_med = float(np.median(Hf)) if Hf.size else 0.0
    S_med = float(np.median(Sf)) if Sf.size else 0.0
    V_med = float(np.median(Vf)) if Vf.size else 0.0
    return H_med, S_med, V_med, frac_light


def predict_single_image(pil_img: Image.Image):
    """
    Run the trained multitask model on a single PIL image.

    Enhancements:
      - Test-Time Augmentation (TTA) + logits averaging
      - Class-prior correction on FABRIC logits
      - Mild temperature sharpening on FABRIC logits
      - Wool/Silk/Synthetic-aware consistency rules using wash-cycle head
      - Robust HSV lightness rule (background-invariant) for COLOR→LIGHT override

    Smart gate logic:
      - If p(is_clothing) < 0.55 -> No garment.
      - OR if both color & fabric confidences are very low (< 0.35),
        we also treat it as non-garment (phone, face, random object).
    """
    x = _tta_batch_from_pil(pil_img)  # [T, C, H, W]

    with torch.no_grad():
        logits_c_list, logits_f_list, logits_w_list, logits_is_list = [], [], [], []
        for t in range(x.size(0)):
            lc, lf, lw, lis = model(x[t].unsqueeze(0))
            logits_c_list.append(lc)
            logits_f_list.append(lf)
            logits_w_list.append(lw)
            logits_is_list.append(lis)

        # Average logits across TTA
        logits_c = torch.mean(torch.stack(logits_c_list, dim=0), dim=0)   # [1, Cc]
        logits_f = torch.mean(torch.stack(logits_f_list, dim=0), dim=0)   # [1, Cf]
        logits_w = torch.mean(torch.stack(logits_w_list, dim=0), dim=0)   # [1, Cw]
        logits_is = torch.mean(torch.stack(logits_is_list, dim=0), dim=0) # [1, 2]

        # Prior correction on FABRIC (Bayes de-biasing)
        if fabric_log_priors.numel() == logits_f.shape[1]:
            logits_f = logits_f - PRIOR_ALPHA * fabric_log_priors.to(device).unsqueeze(0)

        # Mild temperature sharpening for FABRIC
        FABRIC_TEMPERATURE = 0.9  # <1 sharpens slightly
        logits_f = logits_f / FABRIC_TEMPERATURE

        # Convert to probabilities
        probs_c  = F.softmax(logits_c, dim=1)
        probs_f  = F.softmax(logits_f, dim=1)
        probs_w  = F.softmax(logits_w, dim=1)
        probs_is = F.softmax(logits_is, dim=1)  # [1, 2]

        # Argmax + confidences
        max_pc, idx_c = probs_c.max(dim=1)
        max_pf, idx_f = probs_f.max(dim=1)
        max_pw, idx_w = probs_w.max(dim=1)

        max_pc = max_pc.item()
        max_pf = max_pf.item()
        max_pw = max_pw.item()
        p_is_cloth = probs_is[0, 1].item()

        # --------- SMART NON-GARMENT GATE ----------
        if (p_is_cloth < 0.55) or (max_pc < 0.35 and max_pf < 0.35):
            return {
                "color":  "No garment detected",
                "fabric": "No garment detected",
                "wash":   "No washing program suggested — "
                          "the image does not appear to contain clothing.",
            }

        # Low-confidence flag for garment predictions
        LOW_CONF = 0.55
        low_conf_flag = (min(max_pc, max_pf, max_pw) < LOW_CONF) or (p_is_cloth < 0.70)

        # IDs
        pc = idx_c.item()
        pf = idx_f.item()
        pw = idx_w.item()

        # --- Fabric top-2 for post-rules ---
        topk = min(2, probs_f.shape[1])
        top2_vals, top2_idx = torch.topk(probs_f[0], k=topk)
        pf1_id  = int(top2_idx[0].item())
        pf2_id  = int(top2_idx[1].item()) if topk > 1 else pf1_id
        pf1_p   = float(top2_vals[0].item())
        pf2_p   = float(top2_vals[1].item()) if topk > 1 else pf1_p

        wash_key = wash_map.get(pw, f"wash_{pw}")

        # --- Wool-aware rule: if wash-cycle suggests delicate/wool and WOOL is close second, prefer WOOL ---
        woolish_cycle = isinstance(wash_key, str) and (
            ("wool" in wash_key.lower()) or ("delicate" in wash_key.lower()) or ("hand" in wash_key.lower())
        )
        if wool_label_id is not None and woolish_cycle:
            EPS_WOOL = 0.10
            if pf1_id != wool_label_id and (pf2_id == wool_label_id) and ((pf1_p - pf2_p) <= EPS_WOOL):
                pf = wool_label_id
                max_pf = pf2_p  # for any downstream use

        # --- Silk-aware rule: delicate/hand-wash implies silk if close second ---
        silkish_cycle = isinstance(wash_key, str) and (
            ("silk" in wash_key.lower()) or ("delicate" in wash_key.lower()) or ("hand" in wash_key.lower())
        )
        if silk_label_id is not None and silkish_cycle:
            EPS_SILK = 0.08  # slightly stricter than wool
            if pf1_id != silk_label_id and (pf2_id == silk_label_id) and ((pf1_p - pf2_p) <= EPS_SILK):
                pf = silk_label_id
                max_pf = pf2_p

        # --- Synthetic-aware rule: synthetic/easy-care implies synthetic if close second ---
        synthetic_cycle = isinstance(wash_key, str) and (
            ("synthetic" in wash_key.lower()) or ("easy-care" in wash_key.lower()) or ("easycare" in wash_key.lower())
        )
        if synthetic_label_id is not None and synthetic_cycle:
            EPS_SYN = 0.10
            if pf1_id != synthetic_label_id and (pf2_id == synthetic_label_id) and ((pf1_p - pf2_p) <= EPS_SYN):
                pf = synthetic_label_id
                max_pf = pf2_p

    color_name  = color_map.get(pc,  f"Unknown (id={pc})")
    fabric_name = fabric_map.get(pf, f"Unknown (id={pf})")

    wash_key = wash_map.get(pw, f"wash_{pw}")
    full_wash_text = wash_full_description.get(wash_key, wash_key)

    # ============================================================
    # COLOR LIGHT RULE (robust, background-invariant)
    # Only flip to LIGHT if a large fraction of garment pixels are pastel-like.
    # ============================================================
    H_med, S_med, V_med, frac_light = compute_hsv_stats_masked(pil_img)

    # Requirements to override to LIGHT
    RATIO_THRESH   = 0.60   # at least 60% of garment pixels are very light
    S_MED_MAX      = 0.25   # median saturation must be low
    V_MED_MIN      = 0.82   # median brightness must be high

    if (frac_light >= RATIO_THRESH) and (S_med <= S_MED_MAX) and (V_med >= V_MED_MIN):
        color_name = "LIGHT"

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
            # ignore logging errors (read-only FS, etc.)
            pass

        st.success("Prediction done.")
    else:
        st.info("Upload a garment image to receive an automatic washing recommendation.")


if __name__ == "__main__":
    main()
