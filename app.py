def predict_single_image(pil_img: Image.Image):
    """
    Strong garment detection filter:
      - Rejects phone cases, faces, objects.
      - Accepts real garments (swimsuit, shirt, glove).
    """

    # --------------------------
    # 0) Image texture analysis
    # --------------------------
    import numpy as np

    img_arr = np.array(pil_img.resize((128, 128))).astype(np.float32)
    color_std = img_arr.std()   # low std => smooth => probably NOT cloth

    # --------------------------
    # 1) Model forward
    # --------------------------
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w, logits_is = model(x)

        probs_c  = F.softmax(logits_c, dim=1)
        probs_f  = F.softmax(logits_f, dim=1)
        probs_w  = F.softmax(logits_w, dim=1)
        probs_is = F.softmax(logits_is, dim=1)

        max_pc, idx_c = probs_c.max(dim=1)
        max_pf, idx_f = probs_f.max(dim=1)
        max_pw, idx_w = probs_w.max(dim=1)

        max_pc = max_pc.item()
        max_pf = max_pf.item()
        max_pw = max_pw.item()

        p_is_cloth = probs_is[0, 1].item()

    # --------------------------
    # 2) STRONG FILTER RULE
    # --------------------------
    # Condition A: Confidence of being clothing
    cond_A = p_is_cloth >= 0.65

    # Condition B: Some clear structure (texture) exists
    cond_B = (max_pc >= 0.45) or (max_pf >= 0.45)

    # Condition C: Image not too smooth → not phone case
    cond_C = (color_std >= 18)

    if not (cond_A and cond_B and cond_C):
        return {
            "color":  "No garment detected",
            "fabric": "No garment detected",
            "wash":   "No washing program suggested — "
                      "the image does not appear to contain clothing.",
        }

    # --------------------------
    # 3) Garment Prediction
    # --------------------------
    pc = idx_c.item()
    pf = idx_f.item()
    pw = idx_w.item()

    color_name  = color_map.get(pc,  f"Unknown (id={pc})")
    fabric_name = fabric_map.get(pf, f"Unknown (id={pf})")

    wash_key = wash_map.get(pw, f"wash_{pw}")
    full_wash = wash_full_description.get(wash_key, wash_key)

    # Confidence badge
    LOW_CONF = 0.55
    if min(max_pc, max_pf, max_pw, p_is_cloth) < LOW_CONF:
        full_wash = "[Low confidence] " + full_wash

    return {
        "color":  color_name,
        "fabric": fabric_name,
        "wash":   full_wash,
    }
