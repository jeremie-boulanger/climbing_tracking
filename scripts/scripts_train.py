import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# ============================================================
# Configuration
# ============================================================
LIMBS = ['LH', 'RH', 'LF', 'RF']

# post-processing
MIN_LEN_FRAMES = 3     # ~0.06 s at 50 Hz (was 20 = 0.4 s, killed real
                       # 0.08-0.2 s contacts)
ON_THR = 0.6           # hysteresis: turn ON above this
OFF_THR = 0.3          # hysteresis: turn OFF below this
MAX_DIST_M = 0.15      # distance gate: no contact possible beyond this

# training
NEG_POS_RATIO = 10
LR = 1e-3
EPOCHS_DEFAULT = 4000
VAL_FRACTION = 0.2     # last block of each series held out for tuning


# ============================================================
# Annotations (validated state machine)
# ============================================================
def get_annotation(annotation_file, n, list_holds, time, verbose=True):
    """Binary on/off series per (limb, hold) from A/D events.

    Validations (each issue is reported instead of silently corrupting
    the labels):
    - events are processed in TIME order (annotation rows can be
      unsorted; a 'D' beyond the video end processed after a later 'A'
      used to leave the hold occupied until the end of the session);
    - 'A' on hold '-1' (unlabeled support) closes the hold that limb
      was on;
    - 'A' while the limb is already on another hold -> the previous
      hold is implicitly closed (a limb can be on only one hold;
      missing 'D' events used to produce phantom occupations of up
      to 49% of a session);
    - 'D' without a matching 'A' -> skipped;
    - holds not present in the mask folder -> skipped;
    - still occupied at the end -> closed at the last frame.
    """
    if annotation_file.endswith('.xlsx'):
        annotation = pd.read_excel(annotation_file)
    elif annotation_file.endswith('.csv'):
        annotation = pd.read_csv(annotation_file)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")

    def _parse_t(t):
        t = str(t).strip()
        if ':' in t:                     # MM:SS.SS or HH:MM:SS.SS
            head, s = t.rsplit(':', 1)
            if ':' in head:              # HH:MM:SS.SS
                h, m = head.split(':')
                return int(float(h)) * 3600 + int(float(m)) * 60 + float(s)
            return int(float(head)) * 60 + float(s)
        return float(t)                  # bare seconds, e.g. "0"

    time_annotation = [_parse_t(t) for t in annotation["Time"]]
    # Rows are processed in TIME order: annotation files are not always
    # sorted, and an out-of-range 'D' processed after a later-in-time 'A'
    # used to leave the hold occupied until the end of the session.
    order = np.argsort(time_annotation, kind="stable")
    rows = [(time_annotation[k], k, annotation.iloc[k]) for k in order]

    annotation_ts = {limb: {hold: np.zeros(n) for hold in list_holds} for limb in LIMBS}
    hold_states = {limb: {hold: 0 for hold in list_holds} for limb in LIMBS}
    warnings = []

    prev_t = -np.inf
    for t, k, row in rows:
        limb = row["Member parts"]
        hold_id = str(row["Holds ID"]).strip()
        action = row["Action"]

        if t < prev_t:
            warnings.append(f"Row {k}: time {t} not increasing (previous "
                            f"{prev_t}) — check the CSV.")
        prev_t = t

        if limb not in LIMBS:
            continue
        if hold_id == "-1":
            # 'A' on -1 = the limb moved to unlabeled support: whatever
            # hold that limb was on is released.
            if action == "A":
                for h2, st in hold_states[limb].items():
                    if st == 1:
                        i = np.argmin(np.abs(time - t))
                        annotation_ts[limb][h2][i] = -1
                        hold_states[limb][h2] = 0
                        warnings.append(f"Row {k}: {limb} moved to unlabeled "
                                        f"support at t={t}s — closing {h2} "
                                        f"(missing 'D').")
            continue
        if hold_id not in list_holds:
            if verbose:
                warnings.append(f"Row {k}: hold '{hold_id}' not in mask "
                                f"folder — skipped.")
            continue

        i = np.argmin(np.abs(time - t))
        if action == "A" and hold_states[limb][hold_id] == 0:
            for h2, st in hold_states[limb].items():  # implicit close
                if st == 1:
                    annotation_ts[limb][h2][i] = -1
                    hold_states[limb][h2] = 0
                    warnings.append(f"Row {k}: {limb} takes {hold_id} at "
                                    f"t={t}s while still on {h2} — closing "
                                    f"{h2} (missing 'D').")
            annotation_ts[limb][hold_id][i] = 1
            hold_states[limb][hold_id] = 1
        elif action == "D" and hold_states[limb][hold_id] == 1:
            annotation_ts[limb][hold_id][i] = -1
            hold_states[limb][hold_id] = 0
        else:
            warnings.append(f"Row {k}: unmatched {action} on {limb}/"
                            f"{hold_id} at t={t}s — skipped.")

    for limb in annotation_ts:
        for hold in annotation_ts[limb]:
            if hold_states[limb][hold] == 1:
                annotation_ts[limb][hold][-1] = -1
                warnings.append(f"{limb}/{hold}: still occupied at the end "
                                f"— closing at the last frame.")

    for w in warnings:
        print(f"[annotation {os.path.basename(annotation_file)}] {w}")

    # A/D events -> binary occupancy
    for limb in annotation_ts:
        for hold in annotation_ts[limb]:
            annotation_ts[limb][hold] = np.clip(
                np.cumsum(annotation_ts[limb][hold]), 0, 1)
    return annotation_ts


# ============================================================
# Tracking / masks / features
# ============================================================
def get_pos(tracking_file):
    """Per-limb pixel positions + visibility + time from the tracking
    pickle. Visibility is copied through (needed as a feature)."""
    with open(tracking_file, 'rb') as o:
        data = pickle.load(o)
    pos = {}
    for tp in ["RH", "LH", "RF", "LF", "H"]:
        n = data["Time (s)"].shape[0]
        p = np.zeros((n, 2))
        p[:, 0] = data[tp + "x (pix)"]
        p[:, 1] = data[tp + "y (pix)"]
        pos[tp] = p
        key = tp + " visib"
        pos[key] = (np.asarray(data[key], dtype=float).copy()
                    if key in data else np.zeros(n))
    pos['time'] = data["Time (s)"].copy()
    return pos, data["dx"], data["dy"]


def get_mask_holds(folder_holds):
    """Load hold masks -> distance transforms (0 inside the hold)."""
    mask_holds = {}
    for f in os.listdir(folder_holds):
        if "hold_" in f:
            h = f.split('_')[1].replace(".png", "")
            img = cv2.imread(os.path.join(folder_holds, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask_holds[h] = cv2.distanceTransform(1 - thresh, cv2.DIST_L2, 0)
    if mask_holds:
        h0 = list(mask_holds.keys())[0]
        print(f"[masks] {len(mask_holds)} holds, mask size "
              f"{mask_holds[h0].shape[1]}x{mask_holds[h0].shape[0]} (WxH) — "
              f"must match the video frames for the geometry to be right")
    return mask_holds


def get_annotation_hold_ids(annotation_file):
    """Hold IDs referenced by an annotation file (strings, no '-1')."""
    if annotation_file.endswith('.xlsx'):
        annotation = pd.read_excel(annotation_file)
    elif annotation_file.endswith('.csv'):
        annotation = pd.read_csv(annotation_file)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")
    ids = []
    for h in annotation["Holds ID"].astype(str):
        h = h.strip()
        if h not in ids and h not in ("-1", "nan", ""):
            ids.append(h)
    return ids


def build_aligned_masks(sessions, mask_out, radius_min=18, radius_max=40,
                        margin=8, verbose=True):
    """Rebuild hold masks at the TRUE hold locations, from the data.

    Hand-drawn masks can be misaligned with the video frames (drawn on
    another view, resized image, wrong session...). The tracked limb
    positions during ANNOTATED contacts give the true hold locations in
    video pixels (the tracking itself is verified against the frames),
    so a mask built there matches the geometry the labels describe.

    For each hold ID found in the annotations, contact positions are
    pooled across the given sessions and limbs; the mask is a disk
    centred on the median position, radius = p90 + margin (clamped),
    written to <mask_out>/hold_<id>.png. Point the 'mask' field of the
    session configs at <mask_out> afterwards.

    Returns {hold: (cx, cy, radius, n_contact_frames)}.
    """
    pooled = {}        # hold -> list of (n,2) position arrays
    per_session = {}   # hold -> {session index: (center, n)}
    size = None
    for idx, s in enumerate(sessions):
        list_pos, dx, dy = get_pos(s["position"].strip())
        time = list_pos['time']
        n = time.shape[0]

        if size is None and s.get("video"):
            cap = cv2.VideoCapture(s["video"].strip())
            if cap.isOpened():
                size = (int(cap.get(4)), int(cap.get(3)))  # (h, w)
            cap.release()

        hold_ids = get_annotation_hold_ids(s["annotation"].strip())
        annot = get_annotation(s["annotation"].strip(), n, hold_ids, time)
        for hold in hold_ids:
            pts = []
            for tp in LIMBS:
                occ = annot[tp].get(hold)
                if occ is None:
                    continue
                sel = occ > 0
                if sel.any():
                    pts.append(list_pos[tp][sel])
            if not pts:
                continue
            pts = np.concatenate(pts)
            pooled.setdefault(hold, []).append(pts)
            per_session.setdefault(hold, {})[idx] = (np.median(pts, axis=0),
                                                     len(pts))

    if size is None:
        for s in sessions:
            folder = s.get("mask", "").strip()
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if "hold_" in f:
                        img = cv2.imread(os.path.join(folder, f),
                                         cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            size = img.shape[:2]
                            break
            if size is not None:
                break
    if size is None:
        size = (1080, 1920)

    os.makedirs(mask_out, exist_ok=True)
    info = {}
    for hold, pts_list in pooled.items():
        pts = np.concatenate(pts_list)
        center = np.median(pts, axis=0)
        for idx, (c, nf) in per_session.get(hold, {}).items():
            if np.linalg.norm(c - center) > 60:
                print(f"[masks] WARNING: hold {hold} in session {idx} is "
                      f"at ({c[0]:.0f},{c[1]:.0f}) but the pooled center "
                      f"is ({center[0]:.0f},{center[1]:.0f}) — different "
                      f"camera? Split the sessions and rebuild per camera.")
        d = np.linalg.norm(pts - center, axis=1)
        r = int(np.clip(np.percentile(d, 90) + margin,
                        radius_min, radius_max))
        img = np.zeros(size, dtype=np.uint8)
        cv2.circle(img, (int(round(center[0])), int(round(center[1]))),
                   r, 255, -1)
        out = os.path.join(mask_out, f"hold_{hold}.png")
        cv2.imwrite(out, img)
        info[hold] = (center[0], center[1], r, len(pts))
        if verbose:
            print(f"[masks] hold {hold}: center ({center[0]:.0f},"
                  f"{center[1]:.0f}) radius {r}px "
                  f"from {len(pts)} contact frames -> {out}")
    return info


def get_features(list_pos, tp, mask_holds, dx, dy):
    """Features per (limb, hold): [dist_m, vx, vy, speed, |acc|, visib].
    All in consistent units (meters, m/s); z-scored by the network."""
    time = list_pos['time']
    n = time.shape[0]
    dt = np.median(np.diff(time))

    p = list_pos[tp]                       # pixels
    p_m = p.copy()
    p_m[:, 0] *= float(np.asarray(dx).ravel()[0])
    p_m[:, 1] *= float(np.asarray(dy).ravel()[0])
    visib = list_pos[tp + " visib"]

    velo = np.diff(p_m, axis=0) / dt
    velo = np.concatenate((velo[0:1, :], velo))
    speed = np.linalg.norm(velo, axis=1, keepdims=True)
    acc = np.diff(velo, axis=0) / dt
    acc = np.concatenate((acc[0:1, :], acc))
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)

    feat = {}
    for l in mask_holds:
        img = mask_holds[l]
        iy = np.clip(p[:, 1].astype(int), 0, img.shape[0] - 1)
        ix = np.clip(p[:, 0].astype(int), 0, img.shape[1] - 1)
        dist_m = img[iy, ix][:, np.newaxis] * float(np.asarray(dx).ravel()[0])
        feat[l] = np.concatenate((dist_m, velo, speed, acc_mag,
                                  visib[:, np.newaxis]), axis=1)
    return feat


def get_dataset(sessions, dataset_file):
    x, y, m, i = [], [], [], []
    for idx, s in enumerate(sessions):
        list_pos, dx, dy = get_pos(s["position"].strip())
        time = list_pos['time']
        n = time.shape[0]
        mask_holds = get_mask_holds(s["mask"].strip())
        annot = get_annotation(s["annotation"].strip(), n,
                               list(mask_holds.keys()), time)
        for tp in LIMBS:
            feats = get_features(list_pos, tp, mask_holds, dx, dy)
            for l in mask_holds:
                x.append(feats[l])
                y.append(annot[tp][l])
                m.append(np.full(n, f"{tp} - Prise {l}"))
                i.append(np.full(n, idx))

    x = np.concatenate(x).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)
    m = np.concatenate(m)
    i = np.concatenate(i)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0

    os.makedirs(os.path.dirname(dataset_file) or ".", exist_ok=True)
    with open(dataset_file, 'wb') as handle:
        pickle.dump({'x': x, 'y': y, 'm': m, 'i': i,
                     'mean': mean, 'std': std}, handle)
    return


# ============================================================
# Network — z-scoring inside the module
# ============================================================
class u_neural(nn.Module):
    def __init__(self, X=None, Y=None, layers=(64, 16), n_features=6):
        super().__init__()
        self.X, self.Y = X, Y
        n_in = X.shape[1] if X is not None else n_features
        if X is not None:
            mean = X.mean(dim=0)
            std = X.std(dim=0)
            std[std < 1e-8] = 1.0
        else:
            mean, std = torch.zeros(n_in), torch.ones(n_in)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        mod = [nn.Linear(n_in, layers[0]), nn.ReLU()]
        for k in range(len(layers) - 1):
            mod += [nn.Linear(layers[k], layers[k + 1]), nn.ReLU()]
        mod += [nn.Linear(layers[-1], 1)]
        self.net = nn.Sequential(*mod)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def proba(self, x):
        return torch.sigmoid(self.forward(x))

    def loss(self, pos_weight=1.0):
        logits = self(self.X)
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=logits.device)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, self.Y, pos_weight=pw)


def remove_short_sequences(x, min_len=MIN_LEN_FRAMES):
    """Remove 1-valued runs shorter than min_len frames."""
    if len(x) == 0:
        return x
    x = np.asarray(x).copy()
    diff = np.diff(np.concatenate(([0], x, [0])))
    on = np.where(diff == 1)[0]
    off = np.where(diff == -1)[0]
    for a, b in zip(on, off):
        if (b - a) < min_len:
            x[a:b] = 0
    return x


def hysteresis(proba, on_thr=ON_THR, off_thr=OFF_THR):
    """Binarize a probability signal with hysteresis."""
    out = np.zeros(len(proba), dtype=int)
    state = 0
    for t, p in enumerate(proba):
        if state == 0 and p > on_thr:
            state = 1
        elif state == 1 and p < off_thr:
            state = 0
        out[t] = state
    return out


def _postprocess_series(proba, dist, on_thr, off_thr, max_dist_m,
                        min_len=MIN_LEN_FRAMES):
    """Hysteresis -> distance gate -> short-run removal (one series)."""
    state = hysteresis(proba, on_thr, off_thr)
    state[dist > max_dist_m] = 0
    return remove_short_sequences(state, min_len)


def _metrics(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, 1)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 1)
    tp = float((y_true * y_pred).sum())
    fp = float(((1 - y_true) * y_pred).sum())
    fn = float((y_true * (1 - y_pred)).sum())
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall


def _block_split(Y_all, m, val_fraction=VAL_FRACTION):
    """Hold out the last block of each series (random splits leak)."""
    train_idx, val_idx = [], []
    for series in np.unique(m):
        s_idx = np.where(m == series)[0]
        cut = int(len(s_idx) * (1 - val_fraction))
        train_idx.append(s_idx[:cut])
        val_idx.append(s_idx[cut:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


# ============================================================
# Training (with threshold tuning on the held-out block)
# ============================================================
def _train_model(X_train, Y_train, layers, epochs, device):
    model = u_neural(X_train, Y_train, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_pos = Y_train.sum().clamp(min=1)
    n_neg = (1 - Y_train).sum()
    pw = float((n_neg / n_pos).clamp(1, 50).item())
    for _ in range(epochs):
        optimizer.zero_grad()
        model.loss(pos_weight=pw).backward()
        optimizer.step()
    model.eval()
    return model


def _tune_thresholds(model, X_val, Y_val, m_val, device):
    """Grid search (on_thr, off_thr, max_dist) maximizing F1 on the
    held-out block. Features are RAW (the model z-scores internally)."""
    with torch.no_grad():
        p = model.proba(torch.tensor(X_val, dtype=torch.float32,
                                     device=device)).cpu().numpy().flatten()
    dist = X_val[:, 0]
    best = None
    for on_thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for off_thr in [0.2, 0.3, 0.4, 0.5]:
            if off_thr >= on_thr:
                continue
            for max_dist in [0.05, 0.10, 0.15, 0.25]:
                y_pred = np.zeros_like(Y_val)
                for series in np.unique(m_val):
                    sel = np.where(m_val == series)[0]
                    y_pred[sel] = _postprocess_series(
                        p[sel], dist[sel], on_thr, off_thr, max_dist)
                f1, precision, recall = _metrics(Y_val, y_pred)
                if best is None or f1 > best["f1"]:
                    best = {"f1": f1, "precision": precision,
                            "recall": recall, "on_thr": on_thr,
                            "off_thr": off_thr, "max_dist_m": max_dist}
    return best


def train_NN(dataset_file, network_file, layers=(64, 16), display=True,
             epochs=EPOCHS_DEFAULT, tune=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(dataset_file, 'rb') as o:
        dataset = pickle.load(o)

    X_all, Y_all, m_all = dataset["x"], dataset["y"], dataset["m"]

    if tune:
        train_idx, val_idx = _block_split(Y_all, m_all)
    else:
        train_idx = np.arange(len(Y_all))
        val_idx = np.array([], dtype=int)

    # subsample negatives on the training part only
    Y_tr = Y_all[train_idx]
    pos_idx = train_idx[np.where(Y_tr == 1)[0]]
    zero_idx = train_idx[np.where(Y_tr == 0)[0]]
    n_zero = min(len(zero_idx), len(pos_idx) * NEG_POS_RATIO)
    idx = np.concatenate([pos_idx, np.random.choice(zero_idx, size=n_zero,
                                                    replace=False)])
    np.random.shuffle(idx)
    X_train = torch.tensor(X_all[idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_all[idx, None], dtype=torch.float32, device=device)

    model = _train_model(X_train, Y_train, layers, epochs, device)

    if tune and len(val_idx) > 0:
        best = _tune_thresholds(model, X_all[val_idx], Y_all[val_idx],
                                 m_all[val_idx], device)
        try:
            st.info(f"Tuned on validation: ON>{best['on_thr']}, "
                    f"OFF<{best['off_thr']}, gate {best['max_dist_m']}m "
                    f"→ F1 {best['f1']:.2f}")
        except Exception:
            pass
    else:
        best = {"on_thr": ON_THR, "off_thr": OFF_THR, "max_dist_m": MAX_DIST_M}

    # honest report on the full dataset, natural class balance
    with torch.no_grad():
        p_all = model.proba(torch.tensor(X_all, dtype=torch.float32,
                                         device=device)).cpu().numpy().flatten()
    y_pred = np.zeros_like(Y_all)
    for series in np.unique(m_all):
        sel = np.where(m_all == series)[0]
        y_pred[sel] = _postprocess_series(p_all[sel], X_all[sel, 0],
                                          best["on_thr"], best["off_thr"],
                                          best["max_dist_m"])
    f1, precision, recall = _metrics(Y_all, y_pred)

    fig = None
    if display:
        fig = plt.figure()
        plt.bar(["F1", "Precision", "Recall"], [f1, precision, recall])
        plt.ylim(0, 1)
        plt.grid(axis="y")
        try:
            st.pyplot(fig)
        except Exception:
            pass

    os.makedirs(os.path.dirname(network_file) or ".", exist_ok=True)
    torch.jit.script(model).save(network_file)
    params_file = os.path.splitext(network_file)[0] + "_params.json"
    with open(params_file, 'w') as f:
        json.dump({"on_thr": best["on_thr"], "off_thr": best["off_thr"],
                   "max_dist_m": best["max_dist_m"],
                   "min_len_frames": MIN_LEN_FRAMES}, f, indent=2)
    return [f1, precision, recall], fig

# ============================================================
# Inference
# ============================================================
def get_detection_one(list_pos, mask_holds, u_net, dx, dy, device=None,
                      on_thr=ON_THR, off_thr=OFF_THR, max_dist_m=MAX_DIST_M):
    """Detection for one session: {limb: {hold: binary array}}.
    Hysteresis + distance gate per (limb, hold), then winner-take-all
    (a limb can be on only one hold)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = list_pos['time'].shape[0]
    y0 = {}
    for tp in LIMBS:
        feats = get_features(list_pos, tp, mask_holds, dx, dy)
        proba = np.zeros((n, len(mask_holds)))
        for en, l in enumerate(mask_holds):
            X = torch.tensor(feats[l], dtype=torch.float32, device=device)
            with torch.no_grad():
                proba[:, en] = torch.sigmoid(u_net(X)).cpu().numpy().flatten()
        states = np.zeros_like(proba)
        for en in range(len(mask_holds)):
            s = hysteresis(proba[:, en], on_thr, off_thr)
            s[feats[list(mask_holds.keys())[en]][:, 0] > max_dist_m] = 0
            states[:, en] = s
        for t in range(n):  # winner-take-all among active holds
            on_cols = np.where(states[t] > 0)[0]
            if len(on_cols) > 1:
                best = on_cols[np.argmax(proba[t, on_cols])]
                states[t, on_cols] = 0
                states[t, best] = 1
        y0[tp] = {}
        for en, l in enumerate(mask_holds):
            y0[tp][l] = remove_short_sequences(states[:, en].astype(int))
    return y0


def get_detection(sessions, dataset_out, u_net_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, y_score_true, y_score_pred = [], [], [], []
    color = {'LH': 'red', 'RH': 'orange', 'LF': 'blue', 'RF': 'purple'}

    u_net = torch.jit.load(u_net_file, map_location=device)
    u_net.eval()
    tuned = {"on_thr": ON_THR, "off_thr": OFF_THR, "max_dist_m": MAX_DIST_M}
    params_file = os.path.splitext(u_net_file)[0] + "_params.json"
    if os.path.isfile(params_file):
        with open(params_file) as f:
            tuned.update(json.load(f))

    bar = st.progress(0., text="Inference progress")
    annot = None
    for i, s in enumerate(sessions):
        bar.progress(i / len(sessions), text="Inference progress")
        list_pos, dx, dy = get_pos(s["position"].strip())
        time = list_pos['time']
        mask_holds = get_mask_holds(s["mask"].strip())
        n = time.shape[0]

        fig = s.get("plot", None)
        if "annotation" in s:
            annot = get_annotation(s["annotation"].strip(), n,
                                   list(mask_holds.keys()), time)
        if fig is not None:
            plt.figure(figsize=(10, max(len(mask_holds), 1)))
            plt.tight_layout()

        detection_out = {"Time": time}
        y0 = get_detection_one(list_pos, mask_holds, u_net, dx, dy, device,
                               tuned["on_thr"], tuned["off_thr"],
                               tuned["max_dist_m"])
        for tp in LIMBS:
            feats = get_features(list_pos, tp, mask_holds, dx, dy)
            detection_out[tp] = {}
            for en, l in enumerate(mask_holds):
                y_det = y0[tp][l].flatten()
                x.append(feats[l])
                y.append(y_det)
                detection_out[tp][l] = y_det.copy()
                if fig is not None:
                    plt.subplot(len(mask_holds), 1, en + 1)
                    plt.plot(time, y_det, c=color[tp], ls='solid', label=tp)
                    if annot is not None:
                        plt.plot(time, annot[tp][l], c=color[tp], ls='dotted')
                    plt.ylim(0, 1.1)
                    plt.ylabel("Hold:" + str(l))
                if annot is not None:
                    y_score_true.append(annot[tp][l].flatten())
                    y_score_pred.append(y_det)

        if fig is not None:
            if annot is not None:
                yt = np.clip(np.concatenate(y_score_true), 0, 1)
                yp = np.concatenate(y_score_pred)
                f1, precision, recall = _metrics(yt, yp)
                plt.title(f"F1: {f1:.2f}  P: {precision:.2f}  R: {recall:.2f}")
            os.makedirs(os.path.dirname(fig) or ".", exist_ok=True)
            plt.savefig(fig)

        pred_path = s.get("prediction", None)
        if pred_path:
            pred_path = pred_path.strip()
            os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
            with open(pred_path, 'wb') as handle:
                pickle.dump(detection_out, handle)

    bar.progress(1., text="Inference progress")

    x = np.concatenate(x)
    y = np.concatenate(y)
    os.makedirs(os.path.dirname(dataset_out) or ".", exist_ok=True)
    with open(dataset_out, 'wb') as handle:
        pickle.dump({'x': x, 'y': y}, handle)

    if annot is None:
        return 1
    y_true = np.clip(np.concatenate(y_score_true), 0, 1)
    y_pred = np.concatenate(y_score_pred)
    f1, precision, recall = _metrics(y_true, y_pred)
    try:
        st.success(f"F1: {f1:.2f}  Precision: {precision:.2f}  "
                   f"Recall: {recall:.2f}")
    except Exception:
        pass
    return f1