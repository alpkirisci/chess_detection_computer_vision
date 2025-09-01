#!/usr/bin/env python3
"""
Example script for using the trained chess detection models for inference.

Now uses a robust, corner-based perspective mapping (homography) so pieces
are assigned to the correct squares even when the board is tilted or only
part of the image.
"""

from ultralytics import YOLO
import json
import cv2
import numpy as np
import os
import itertools

def warp_pts(H, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

def best_H_for_4points(pts4, warp_size=800):
    pts4 = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]], dtype=np.float32)
    best = None
    for perm in itertools.permutations(range(4), 4):
        src = pts4[list(perm)]
        H = cv2.getPerspectiveTransform(src, dst)
        if not np.isfinite(H).all() or np.allclose(H, 0):
            continue
        proj = warp_pts(H, src)
        err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        if best is None or err < best[0]:
            best = (err, H, perm)
    if best is None:
        raise ValueError("No valid homography for given 4 points")
    return best  # (err, H, perm)

def cluster_points(points, confs, tol=150.0):
    order = np.argsort(-confs)
    clusters = []
    for idx in order:
        p = points[idx]
        placed = False
        for g in clusters:
            if np.hypot(*(p - g['pt'])) < tol:
                placed = True
                break
        if not placed:
            clusters.append({'pt': p, 'conf': confs[idx]})
    out_pts = np.array([c['pt'] for c in clusters], dtype=np.float32)
    out_conf = np.array([c['conf'] for c in clusters], dtype=np.float32)
    return out_pts, out_conf

def select_best_4(points, warp_size=800):
    n = len(points)
    best = None
    for combo in itertools.combinations(range(n), 4):
        pts4 = points[list(combo)]
        try:
            err, H, perm = best_H_for_4points(pts4, warp_size)
            if best is None or err < best[0]:
                best = (err, H, perm, combo, pts4)
        except Exception:
            continue
    if best is None:
        raise ValueError("Failed to select a valid set of 4 corners")
    return best  # (err, H, perm, combo, pts4)
# Dataset annotations fallback removed to keep inference model-only

def extract_raw_corner_centers(corner_model: YOLO, image_path: str, conf_threshold=0.25):
    res = corner_model(image_path, conf=conf_threshold)
    pts, confs = [], []
    for r in res:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            pts.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            confs.append(float(c))
    if not pts:
        raise ValueError("No corner detections found")
    return np.array(pts, dtype=np.float32), np.array(confs, dtype=np.float32)

def grid_squares(warp_size=800):
    squares = {}
    cell = warp_size // 8
    for r in range(8):
        for c in range(8):
            x1, y1 = c * cell, r * cell
            x2, y2 = (c + 1) * cell, (r + 1) * cell
            pos = chr(ord('a') + c) + str(8 - r)
            squares[pos] = [x1, y1, x2, y2]
    return squares

def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def map_box_to_square_via_bottom_strip(x1, y1, x2, y2, H, squares, warp_size=800, strip_frac=0.25):
    h = max(1.0, y2 - y1)
    y_top = y2 - strip_frac * h
    poly_img = np.array([[x1, y_top], [x2, y_top], [x2, y2], [x1, y2]], dtype=np.float32)
    warped = warp_pts(H, poly_img)
    wx1, wy1 = float(np.min(warped[:, 0])), float(np.min(warped[:, 1]))
    wx2, wy2 = float(np.max(warped[:, 0])), float(np.max(warped[:, 1]))
    wx1 = np.clip(wx1, 0, warp_size - 1); wy1 = np.clip(wy1, 0, warp_size - 1)
    wx2 = np.clip(wx2, 0, warp_size - 1); wy2 = np.clip(wy2, 0, warp_size - 1)
    warped_rect = [wx1, wy1, wx2, wy2]

    best_pos, best_iou = None, -1.0
    for pos, sq in squares.items():
        iou = rect_iou(warped_rect, sq)
        if iou > best_iou:
            best_iou, best_pos = iou, pos

    if best_iou < 1e-6:
        cx, cy = (x1 + x2) * 0.5, y2
        cx_w, cy_w = warp_pts(H, np.array([[cx, cy]], dtype=np.float32))[0]
        cell = warp_size // 8
        c = int(np.clip(cx_w // cell, 0, 7))
        r = int(np.clip(cy_w // cell, 0, 7))
        best_pos = chr(ord('a') + c) + str(8 - r)
        best_iou = 0.0
    return best_pos, best_iou

def detect_chess_pieces(model_path, image_path, corner_model_path=None, conf_threshold=0.25, warp_size=800):
    """
    Detect chess pieces and return positions in JSON format.
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_path: Path to chess board image
        corner_model_path: Optional path to corner detection model
        conf_threshold: Confidence threshold for detections
        warp_size: Size of the canonical, top-down board used for mapping
    
    Returns:
        dict: Chess board state in the required format
    """
    # Load the trained piece detection model
    model = YOLO(model_path)
    
    # Corner model is required
    if not corner_model_path or not os.path.exists(corner_model_path):
        raise FileNotFoundError("Corner detection model is required. Provide a valid 'corner_model_path'.")

    # Determine board homography H
    H = None
    corners_used = None
    corner_method = "none"
    
    # Compute homography from detected corners (required)
    try:
        corner_model = YOLO(corner_model_path)
        raw_pts, raw_confs = extract_raw_corner_centers(corner_model, image_path, conf_threshold=0.25)
        pts_unique, confs_unique = cluster_points(raw_pts, raw_confs, tol=150.0)
        err, H, perm, combo, pts4 = select_best_4(pts_unique, warp_size=warp_size)
        corners_used = pts4.tolist()
        corner_method = "model_detection"
        print(f"✅ Corner model: using 4 corners (reproj err ~ {err:.2f}px)")
    except Exception as _:
        raise RuntimeError("Failed to compute board corners. Corner model is required for mapping.")
    
    # Perform inference
    results = model(image_path, conf=conf_threshold)
    
    # Get class names
    class_names = model.names
    
    # Convert detections to chess positions
    board_state = []
    squares = grid_squares(warp_size=warp_size) if H is not None else None
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Skip empty squares
                if class_names[cls] == 'empty':
                    continue

                # Map to board square
                pos, iou = map_box_to_square_via_bottom_strip(x1, y1, x2, y2, H, squares, warp_size=warp_size, strip_frac=0.25)
                
                # Format piece name
                piece_name = class_names[cls].replace('-', '_')
                
                board_state.append({
                    "position": pos,
                    "piece": piece_name,
                    "confidence": float(conf),
                    "iou": float(iou)
                })
    
    # Keep only the highest-confidence detection per square
    dedup = {}
    for det in board_state:
        k = det["position"]
        if k not in dedup or det["confidence"] > dedup[k]["confidence"]:
            dedup[k] = det
    board_state = list(dedup.values())
    
    out = {
        "board_state": board_state,
        "corner_detection_method": corner_method,
        "accuracy_level": "high" if corner_method == "model_detection" else "medium"
    }
    if corners_used is not None:
        out["corners_used"] = corners_used
    return out

def visualize_detections(model_path, image_path, output_path=None):
    """
    Visualize chess piece detections on the image.
    
    Args:
        model_path: Path to trained YOLO model
        image_path: Path to chess board image
        output_path: Optional path to save annotated image
    """
    model = YOLO(model_path)
    results = model(image_path)
    
    # Plot results
    annotated_img = results[0].plot()
    
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"Annotated image saved to: {output_path}")
    
    return annotated_img

if __name__ == "__main__":
    # Example usage with dual model system
    PIECE_MODEL_PATH = "models/piece_best.pt"          # Piece detection model
    CORNER_MODEL_PATH = "models/corner_best.pt"        # Corner detection model (required)
    IMAGE_PATH = "chess_board_image.jpg"               # Chess board image
    
    try:
        print("🔍 Chess Detection with Dual Model System")
        print("=" * 50)
        
        # Detect chess pieces (corner model required)
        result = detect_chess_pieces(
            model_path=PIECE_MODEL_PATH,
            image_path=IMAGE_PATH, 
            corner_model_path=CORNER_MODEL_PATH
        )
        
        # Print the result
        print(f"\\nDetection Method: {result['corner_detection_method']}")
        print(f"Accuracy Level: {result['accuracy_level']}")
        print(f"Pieces Detected: {len(result['board_state'])}")
        
        print("\\nChess Board State:")
        print(json.dumps({"board_state": result["board_state"]}, indent=2))
        
        # Visualize detections
        visualize_detections(PIECE_MODEL_PATH, IMAGE_PATH, "annotated_chess_board.jpg")
        
        print("\\n✅ Detection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\nMake sure you have:")
        print("1. ✅ Piece detection model (models/piece_best.pt)")
        print("2. ✅ Input chess board image")
        print("3. ✅ ultralytics package (pip install ultralytics)")
        print("4. ✅ Corner detection model (models/corner_best.pt) — REQUIRED")
