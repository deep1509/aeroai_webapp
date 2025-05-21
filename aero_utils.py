# 📦 aero_utils.py (Fixed Version with Working Video + YOLOv5 CLI Support)

import os
import sys
import subprocess
from pathlib import Path
import cv2
import streamlit as st
from ultralytics import YOLO as YOLOv8
from uuid import uuid4



# 📌 Ensure yolov5 local path is in sys.path
YOLOV5_DIR = os.path.join(os.getcwd(), 'yolov5')
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)
    
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
# 📂 Paths

# Absolute path to detect.py
YOLOV5_DETECT_SCRIPT = os.path.join(YOLOV5_DIR, 'detect.py')

persistent_panels = []

# 🚀 Load Models
def load_models(panel_model_path, anomaly_model_path):
    panel_model = YOLOv8(panel_model_path)

    if not Path(anomaly_model_path).exists():
        raise FileNotFoundError(f"Anomaly model not found at {anomaly_model_path}")

    device = select_device('')
    anomaly_model = DetectMultiBackend(anomaly_model_path, device=device)

    return panel_model, anomaly_model

# 🧠 Parse YOLO labels
def parse_yolo_labels(label_file_path, class_map, image_path=None):
    boxes = []
    if image_path and Path(image_path).exists():
        img = cv2.imread(str(image_path))
        img_h, img_w = img.shape[:2]
    else:
        img_h, img_w = 640, 640

    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            cls, xc, yc, w, h = map(float, parts[:5])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append({
                'class_id': int(cls),
                'class_name': class_map.get(int(cls), f'class_{int(cls)}'),
                'bbox': (x1, y1, x2, y2)
            })
    return boxes

# 🔗 IOU logic
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

def is_center_inside(panel_box, anomaly_box):
    ax1, ay1, ax2, ay2 = anomaly_box
    cx = (ax1 + ax2) / 2
    cy = (ay1 + ay2) / 2
    px1, py1, px2, py2 = panel_box
    return px1 <= cx <= px2 and py1 <= cy <= py2

def is_panel_fully_inside_anomaly(panel_box, anomaly_box):
    px1, py1, px2, py2 = panel_box
    ax1, ay1, ax2, ay2 = anomaly_box
    return px1 >= ax1 and py1 >= ay1 and px2 <= ax2 and py2 <= ay2

from collections import defaultdict
import uuid



def link_anomalies_to_panels(panel_boxes, anomaly_boxes, iou_threshold=0.5):
    global persistent_panels
    # Global tracker across frames (reset per video)
    panel_map = {}

    for panel in panel_boxes:
        matched = False
        panel_bbox = panel['bbox']

        # Try to match with existing persistent panels
        for tracked in persistent_panels:
            tracked_id, tracked_bbox = tracked['id'], tracked['bbox']
            iou = calculate_iou(panel_bbox, tracked_bbox)
            if iou > iou_threshold:
                panel_id = tracked_id
                matched = True
                break

        # If not matched, assign new panel ID
        if not matched:
            panel_id = f"Panel_{str(uuid.uuid4())[:8]}"
            persistent_panels.append({'id': panel_id, 'bbox': panel_bbox})

        panel_map[panel_id] = set()

        # Match this panel to anomaly boxes
        for anomaly in anomaly_boxes:
            abox = anomaly['bbox']
            cls = anomaly['class_name']
            if (
                calculate_iou(panel_bbox, abox) > 0.3 or
                is_center_inside(panel_bbox, abox) or
                is_panel_fully_inside_anomaly(panel_bbox, abox)
            ):
                panel_map[panel_id].add(cls)

    # Default to normal if no anomalies found
    for pid in panel_map:
        if not panel_map[pid]:
            panel_map[pid].add('Not Classified')

    return {k: list(v) for k, v in panel_map.items()}


def process_image_file(uploaded_file, panel_model, anomaly_model_path, save_dir="processed"):
    from datetime import datetime
    from uuid import uuid4

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    filename_stem = Path(uploaded_file.name).stem
    temp_path = save_path / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Panel Detection (YOLOv8)
    panel_results = panel_model.predict(
        source=str(temp_path),
        save=True,
        save_txt=True,
        project=str(save_path),
        name=f"panel_{Path(uploaded_file.name).stem}_{uuid4().hex[:6]}",
        exist_ok=True
    )
    panel_output_dir = Path(panel_results[0].save_dir)
    panel_image_candidates = list(panel_output_dir.glob("*.jpg"))
    panel_output_image = panel_image_candidates[0] if panel_image_candidates else None
    st.session_state[f'panel_image_{uploaded_file.name}'] = str(panel_output_image)


    # Anomaly Detection (YOLOv5)
    anomaly_subdir = f"anomaly_{filename_stem}_{uuid4().hex[:6]}"
    try:
        subprocess.run([
            "python", YOLOV5_DETECT_SCRIPT,
            "--weights", str(anomaly_model_path),
            "--source", str(temp_path),
            "--conf", "0.25",
            "--save-txt", "--save-conf",
            "--project", str(save_path),
            "--name", anomaly_subdir,
            "--exist-ok"
        ], check=True)
    except subprocess.CalledProcessError as e:
        st.error("❌ YOLOv5 anomaly detection failed.")
        st.code(e.stderr or str(e))
        raise

    anomaly_output_dir = save_path / anomaly_subdir
    anomaly_image_candidates = list(anomaly_output_dir.glob("*.jpg"))
    anomaly_output_image = anomaly_image_candidates[0] if anomaly_image_candidates else None

    if anomaly_output_image is None:
        st.warning("⚠️ No anomaly image was generated.")
        
    return panel_output_image, anomaly_output_image


from pathlib import Path
import subprocess
from uuid import uuid4
import re
import streamlit as st

def process_video_file(uploaded_file, panel_model, anomaly_model_path, save_dir="processed"):

    global persistent_panels
    persistent_panels = []  # reset before each video run

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    unique_id = uuid4().hex[:6]
    video_path = save_path / f"temp_video_{unique_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Panel detection
    panel_model.predict(
        source=str(video_path),
        save=True,
        save_txt=True,
        conf=0.25,
        #vid_stride=6,
        project=str(save_path),
        name='panel_video',
        exist_ok=True
    )
    panel_output_video = save_path / "panel_video" / video_path.name

    # Anomaly detection using YOLOv5 CLI
    anomaly_subdir = f"anomaly_video_{unique_id}"
    anomaly_output_dir = save_path / anomaly_subdir
    subprocess.run([
        "python", "yolov5/detect.py",
        "--weights", str(anomaly_model_path),
        "--source", str(video_path),
        "--conf", "0.25",
        "--save-txt", "--save-conf",
        #"--vid-stride", "6",
        "--project", str(save_path),
        "--name", anomaly_subdir,
        "--exist-ok"
    ], check=True)

    # Re-encode video for compatibility
    raw_anomaly_video = anomaly_output_dir / video_path.name
    fixed_anomaly_video = raw_anomaly_video.with_name(raw_anomaly_video.stem + "_fixed.mp4")
    if raw_anomaly_video.exists():
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(raw_anomaly_video),
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                str(fixed_anomaly_video)
            ], check=True)
            final_anomaly_video = fixed_anomaly_video
        except subprocess.CalledProcessError:
            st.warning("⚠️ Re-encoding failed. Attempting to use raw output.")
            final_anomaly_video = raw_anomaly_video
    else:
        st.error("❌ Anomaly output video not found.")
        return panel_output_video, None

    # Preview frame
    anomaly_preview_frame = None
    frames = list(anomaly_output_dir.glob("*.jpg"))
    if frames:
        anomaly_preview_frame = frames[0]

    # Parse label files
    panel_class_map = {0: "panel"}
    anomaly_class_map = {0: "cracked", 1: "dusty", 2: "normal"}

    def extract_frame_id(path):
        match = re.search(r'_(\d+)\.txt$', str(path))
        return int(match.group(1)) if match else -1

    panel_label_files = list((save_path / "panel_video" / "labels").glob("*.txt"))
    anomaly_label_files = list((anomaly_output_dir / "labels").glob("*.txt"))

    panel_map = {extract_frame_id(f): f for f in panel_label_files}
    anomaly_map = {extract_frame_id(f): f for f in anomaly_label_files}
    common_frames = sorted(set(panel_map.keys()) & set(anomaly_map.keys()))
    print("▶️ Panel Frames:", sorted(panel_map.keys()))
    print("⚠️ Anomaly Frames:", sorted(anomaly_map.keys()))
    print("✅ Common Frames:", common_frames)


    total_panels = 0
    count_normal = count_dusty = count_cracked = 0
    combined_map = {}
    last_frame_map = None

    for i, frame_num in enumerate(common_frames):
        plabel = panel_map[frame_num]
        alabel = anomaly_map[frame_num]

        panel_boxes = parse_yolo_labels(plabel, panel_class_map)
        anomaly_boxes = parse_yolo_labels(alabel, anomaly_class_map)
        panel_anomaly_map = link_anomalies_to_panels(panel_boxes, anomaly_boxes)

        if panel_anomaly_map != last_frame_map:
            st.session_state[f'panel_anomaly_map_{video_path.stem}_frame{i+1}'] = panel_anomaly_map
            last_frame_map = panel_anomaly_map

        for panel_id, anomalies in panel_anomaly_map.items():
            if panel_id not in combined_map:
                combined_map[panel_id] = set()
            combined_map[panel_id].update(anomalies)

        total_panels += len(panel_anomaly_map)
        for labels in panel_anomaly_map.values():
            for label in labels:
                if label == 'dusty': count_dusty += 1
                elif label == 'cracked': count_cracked += 1
                elif label == 'normal': count_normal += 1

    merged_map = {k: list(v) for k, v in combined_map.items()}
    st.session_state[f'panel_anomaly_map_{video_path.stem}_summary'] = merged_map

    # Stats
    st.session_state['panel_video'] = str(panel_output_video)
    st.session_state['anomaly_video_frame'] = str(anomaly_preview_frame) if anomaly_preview_frame else None
    st.session_state['summary_temp_video'] = {
        'panels': total_panels,
        'dusty': count_dusty,
        'cracked': count_cracked,
        'normal': count_normal
    }

    # Cleanup
    for key in list(st.session_state.keys()):
        if key.startswith(f"panel_anomaly_map_{video_path.stem}_frame"):
            del st.session_state[key]

    st.session_state[f'anomaly_video_{video_path.stem}'] = str(final_anomaly_video)
    st.session_state[f'anomaly_video_frame_{video_path.stem}'] = str(anomaly_preview_frame) if anomaly_preview_frame else None

    return panel_output_video, final_anomaly_video



