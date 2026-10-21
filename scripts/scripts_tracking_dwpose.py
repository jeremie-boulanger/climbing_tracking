import os
import pickle
import numpy as np
import cv2
import streamlit as st

# ---------------------------------------------------------------------------
# DWPose single-pass tracking (ONNX).
#
# Two compact ONNX models (downloaded to <project>/models/dwpose_models
# at first run):
#   - yolox_l.onnx        : person detector (mmdet YOLOX-L, 640x640 input)
#   - dw-ll_ucoco_384.onnx: wholebody pose, 133 COCO-WholeBody keypoints
# Replaces the YOLO + MediaPipe two-model stack with one coherent pipeline
# that also gives hand and foot keypoints (better hold-contact points).
# Models from https://huggingface.co/yzd-v/DWPose (Apache-2.0).
# ---------------------------------------------------------------------------

DET_INPUT = 640
POSE_W, POSE_H = 288, 384          # pose crop size (W x H)
DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
DET_CONF = 0.25                     # person-class confidence
NMS_IOU = 0.5
BOX_EXPAND = 1.25                   # crop expansion around the person box
IOU_MATCH = 0.3                     # greedy tracker association threshold
MAX_MISS = 25                       # frames a track survives without a match
MIN_TRACK_FRAMES = 100              # tracks shorter than this are not climbers
SEP_MIN = 10.0                      # px: L/R composite closer than this =
                                    # collapsed pair (occlusion)

# COCO-WholeBody (wb) -> MediaPipe-pose (mp) landmark index, for the
# landmarks used downstream (hands, hips, feet, plus the main body ones).
WB_TO_MP = {
    0: 0,      # nose
    5: 11, 6: 12,   # shoulders
    7: 13, 8: 14,   # elbows
    11: 23, 12: 24, # hips
    13: 25, 14: 26, # knees
    15: 27, 16: 28, # ankles
    19: 29, 22: 30, # heels (wb: L_heel, R_heel)
    17: 31, 20: 32, # foot index (wb: big toes)
    91: 15, 112: 16,  # wrists (wb hand wrists)
    93: 21, 114: 22,  # thumb MCP
    96: 19, 117: 20,  # index MCP
    108: 17, 129: 18, # pinky MCP
}
N_MP = 33


def _download_models(model_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for f in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            p = hf_hub_download("yzd-v/DWPose", f,
                                local_dir=model_dir)
        paths[f] = p
    return paths


class DWPose:
    """Lazy holder for the two ONNX sessions."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "dwpose_models")
        self.model_dir = model_dir
        self.det = None
        self.pose = None

    def load(self):
        if self.det is None:
            paths = _download_models(self.model_dir)
            providers = ["CPUExecutionProvider"]
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider",
                                  "CPUExecutionProvider"]
            except Exception:
                pass
            import onnxruntime as ort
            self.det = ort.InferenceSession(paths["yolox_l.onnx"],
                                            providers=providers)
            self.pose = ort.InferenceSession(paths["dw-ll_ucoco_384.onnx"],
                                             providers=providers)
        return self

    # ---------------- detection ----------------
    @staticmethod
    def _letterbox(im):
        h, w = im.shape[:2]
        r = DET_INPUT / max(h, w)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((DET_INPUT, DET_INPUT, 3), 114, dtype=np.float32)
        canvas[(DET_INPUT - nh) // 2:(DET_INPUT - nh) // 2 + nh,
               (DET_INPUT - nw) // 2:(DET_INPUT - nw) // 2 + nw] = \
            cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, (DET_INPUT - nw) // 2, (DET_INPUT - nh) // 2

    def detect_persons(self, img):
        """All person boxes (xyxy, original coords) + scores."""
        canvas, r, dx, dy = self._letterbox(img)
        x = ((canvas[:, :, ::-1] - DET_MEAN) / DET_STD
             ).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.det.run(None, {"images": x})[0][0]   # (8400, 85)
        boxes, scores = [], []
        i0 = 0
        for gs, stride in [(80, 8), (40, 16), (20, 32)]:
            n = gs * gs
            s = out[i0:i0 + n]
            gxv = np.tile(np.arange(gs), gs)[:, None]
            gyv = np.repeat(np.arange(gs), gs)[:, None]
            e = np.exp(s[:, :4])                       # ltrb, grid units
            b = np.concatenate([(gxv - e[:, :1]) * stride,
                                 (gyv - e[:, 1:2]) * stride,
                                 (gxv + e[:, 2:3]) * stride,
                                 (gyv + e[:, 3:4]) * stride], 1)
            boxes.append(b)
            scores.append(s[:, 5])                      # cls 'person'
            i0 += n
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)

        keep = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] < DET_CONF:
                break
            if all(_iou_xyxy(boxes[i], boxes[k]) < NMS_IOU for k in keep):
                keep.append(i)
        boxes = (boxes[keep] - [dx, dy, dx, dy]) / r
        return boxes, scores[keep]

    # ---------------- pose ----------------
    def estimate_pose(self, img, box):
        """133 wholebody keypoints (x, y, conf) in original coords."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w = max(box[2] - box[0], 10) * BOX_EXPAND
        h = max(box[3] - box[1], 10) * BOX_EXPAND
        src = np.array([[cx - w / 2, cy - h / 2],
                        [cx + w / 2, cy - h / 2],
                        [cx - w / 2, cy + h / 2]], dtype=np.float32)
        dst = np.array([[0, 0], [POSE_W, 0], [0, POSE_H]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (POSE_W, POSE_H),
                              flags=cv2.INTER_LINEAR)
        inp = ((crop[:, :, ::-1] - POSE_MEAN) / POSE_STD
               ).transpose(2, 0, 1)[None].astype(np.float32)
        simcc_x, simcc_y = self.pose.run(None, {"input": inp})
        kx = np.argmax(simcc_x[0], axis=1) / 2.0    # 2x resolution
        ky = np.argmax(simcc_y[0], axis=1) / 2.0
        conf = np.maximum(simcc_x[0].max(axis=1),
                          simcc_y[0].max(axis=1)).clip(0, 1)
        Minv = cv2.invertAffineTransform(M)
        pts = np.stack([Minv[0, 0] * kx + Minv[0, 1] * ky + Minv[0, 2],
                        Minv[1, 0] * kx + Minv[1, 1] * ky + Minv[1, 2]],
                       axis=1)
        return np.concatenate([pts, conf[:, None]], axis=1)


def _iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _wb_to_mediapipe_pose(kps):
    """COCO-WholeBody (133, 3) -> 33 MediaPipe-pose landmark dicts, so the
    existing extract_tracking works unchanged on the output.

    Occlusion handling: when the model cannot see one side (climber back
    to the camera), DWPose tends to collapse the left and right keypoints
    of a pair onto the same point — a phantom limb standing on the other
    limb's hold. When the L and R composites of a pair are closer than
    SEP_MIN pixels, the weaker side's visibility is scaled by the
    separation so the visibility-weighted filtering in extract_tracking
    suppresses it (same degradation as MediaPipe visibility)."""
    pose = [{'x': 0., 'y': 0., 'z': 0., 'visib': 0.} for _ in range(N_MP)]
    for wb, mp in WB_TO_MP.items():
        x, y, c = kps[wb]
        pose[mp] = {'x': float(x), 'y': float(y), 'z': 0.,
                    'visib': float(c)}

    # mirrored pairs of MediaPipe indices whose composites must stay apart
    pairs = [
        ([15, 21, 19, 17], [16, 22, 20, 18]),   # hands
        ([27, 29, 31], [28, 30, 32]),           # feet
    ]
    for l_idx, r_idx in pairs:
        l_pts = np.array([[pose[i]['x'], pose[i]['y']] for i in l_idx])
        r_pts = np.array([[pose[i]['x'], pose[i]['y']] for i in r_idx])
        if not (l_pts.any() and r_pts.any()):
            continue
        d = float(np.linalg.norm(l_pts.mean(0) - r_pts.mean(0)))
        if d >= SEP_MIN:
            continue
        factor = d / SEP_MIN
        l_conf = np.mean([pose[i]['visib'] for i in l_idx])
        r_conf = np.mean([pose[i]['visib'] for i in r_idx])
        weak, strong = (l_idx, r_idx) if l_conf <= r_conf else (r_idx, l_idx)
        for i in weak:
            pose[i]['visib'] *= factor
    return pose


def dwpose_tracking(video_file, tracking_initial, max_persons=3,
                    display=True):
    """Single pass over the video: detect every person, keep tracks with a
    greedy IoU association, estimate wholebody pose per person per frame,
    then select the climber(s) (the track(s) that climb the highest and
    last long enough) and save a MediaPipe-compatible pose pickle."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.write("Error opening video stream or file")
        raise TypeError
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    dwpose = DWPose().load()
    tracks = {}      # tid -> {'frames': [...], 'kps': [...], 'boxes': [...], 'miss': int}
    next_id = 0

    bar = st.progress(0., text="DWPose tracking progress")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bar.progress(i / max(total, 1), text="DWPose tracking progress")
        ts = (i / fps) * 1000 if fps else 0.

        boxes, scores = dwpose.detect_persons(frame)
        order = np.argsort(scores)[::-1][:max_persons]

        # greedy IoU association with the active tracks
        assigned = set()
        for tid, tr in tracks.items():
            if tr['miss'] > MAX_MISS:
                continue
            best_j, best_iou = None, 0.
            for j in order:
                if j in assigned:
                    continue
                iou = _iou_xyxy(boxes[j], tr['boxes'][-1])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= IOU_MATCH:
                j = best_j
                assigned.add(j)
                tr['frames'].append(i)
                tr['boxes'].append(boxes[j])
                tr['kps'].append(dwpose.estimate_pose(frame, boxes[j]))
                tr['miss'] = 0
            else:
                tr['miss'] += 1
        # new tracks for unmatched detections
        for j in order:
            if j not in assigned:
                tracks[next_id] = {
                    'frames': [i], 'boxes': [boxes[j]],
                    'kps': [dwpose.estimate_pose(frame, boxes[j])],
                    'miss': 0}
                next_id += 1
        i += 1
    bar.empty()
    cap.release()

    # climber selection: tracks long enough, ranked by highest point
    # reached (smallest y of the box top) — the person climbing the
    # highest. The vertical range traveled breaks ties between a
    # climber and a spotter who stands high but never moves.
    candidates = []
    for tid, tr in tracks.items():
        if len(tr['frames']) < MIN_TRACK_FRAMES:
            continue
        tops = [b[1] for b in tr['boxes']]
        bots = [b[3] for b in tr['boxes']]
        candidates.append((min(tops), -(max(bots) - min(tops)), tid,
                           len(tr['frames'])))
    candidates.sort()
    if not candidates:
        raise RuntimeError(
            "No person tracked long enough in the video "
            f"({MIN_TRACK_FRAMES} frames minimum).")
    climber_ids = [tid for _, _, tid, _ in candidates]
    st.write(f"Climber candidates (highest first): "
             f"{[(tid, int(y), n) for y, _, tid, n in candidates]}")
    climber = climber_ids[0]

    # MediaPipe-compatible pose list for the selected climber
    list_pose = []
    tr = tracks[climber]
    for f, kps in zip(tr['frames'], tr['kps']):
        list_pose.append({'timestamp': (f / fps) * 1000 if fps else 0.,
                          'pose': _wb_to_mediapipe_pose(kps),
                          'world pose': [],
                          'size': (width, height)})

    os.makedirs(os.path.dirname(tracking_initial) or ".", exist_ok=True)
    with open(tracking_initial, 'wb') as handle:
        pickle.dump(list_pose, handle)

    if display:
        st.write(f"Tracked climber: track {climber} "
                 f"({len(tr['frames'])} frames with pose, "
                 f"{len(tracks)} person tracks total).")
        st.write(f"Pose pickle saved at {tracking_initial}")
    return list_pose, climber_ids, tracks
