# Chess Piece Detection Project

This project detects chess pieces on real photos of boards and returns their squares (like "e4") in JSON. I trained two YOLOv8 models on the ChessReD2K dataset and combined them so it works even when the board is tilted, partially visible, or not centered.

## Project Overview

- **Input**: Images of chess boards during games
- **Output**: JSON format with piece positions in algebraic notation
- **Method**: YOLOv8 object detection
- **Dataset**: ChessReD2K (2,078 images, 43,149 piece annotations)

## Output Format

```json
{
  "board_state": [
    {
      "position": "a1",
      "piece": "white_rook"
    },
    {
      "position": "g8", 
      "piece": "black_knight"
    }
  ]
}
```

## Files & Structure

```
chess-detection/
├── annotations.json              # Original full dataset annotations
├── chessred2k_annotations.json   # Filtered ChessReD2K annotations
├── chessred2k/                   # ChessReD2K image dataset
│   └── images/
│       ├── 0/
│       ├── 6/
│       ├── 19/
│       └── ...
├── yolo_dataset/                 # Converted YOLO format dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── dataset.yaml
├── filter_annotations.py         # Script to filter annotations
├── convert_to_yolo.py            # Script to convert to YOLO format
├── chess_inference_example.py      # Inference with robust corner-based mapping
└── chess_detection_training.ipynb  # Google Colab training notebook
```

## Datasets (YOLO format)

- **Pieces YOLO dataset**: [Google Drive link](https://drive.google.com/drive/folders/1IMkCi-F_oXUQOYbPu6CLmcugU-qixrHH?usp=sharing)
- **Corners YOLO dataset**: [Google Drive link](https://drive.google.com/drive/folders/1ZfEUz5iiaLxsD8v_erVTvj8yadNA_YcH?usp=sharing)

## What I Built (Simple Explanation)

- **Two models working together**:
  - **Pieces model** (YOLOv8m): finds every piece and its bounding box.
  - **Corners model** (YOLOv8n): finds the 4 corners of the board.
- **Why two models?** If you only detect pieces and split the image into an 8x8 grid, it breaks when the camera is at an angle. Corners let me “unwarp” the board to a flat, top-down view. Then placing pieces on squares is reliable.
- **How mapping works**: from the detected corners, I compute a perspective transform (homography) to a clean 800×800 board. For each piece, I project a small strip at the bottom of its box into that top-down view and pick the overlapping square. This handles perspective and partial boards.

### ChessReD2K Subset
- **Total Images**: 2,078
- **Piece Annotations**: 43,149
- **Corner Annotations**: 2,078
- **Game IDs**: 0, 6, 19, 22, 28, 33, 38, 41, 42, 47, 56, 58, 61, 72, 76, 78, 83, 87, 91, 99

### Chess Piece Classes (13 total)
0. white-pawn
1. white-rook
2. white-knight
3. white-bishop
4. white-queen
5. white-king
6. black-pawn
7. black-rook
8. black-knight
9. black-bishop
10. black-queen
11. black-king
12. empty

### Data Split
- **Train**: 1,454 images (70%)
- **Validation**: 312 images (15%)
- **Test**: 312 images (15%)

## Key Features

### 1. Automated Data Processing
- Filters full ChessReD dataset to ChessReD2K subset
- Converts COCO format to YOLOv8 format automatically
- Handles both piece and corner annotations

### 2. Ready-to-Use Training Pipeline
- Complete Google Colab notebook
- Optimized for A100 GPU
- Includes visualization and evaluation

### 3. Chess-Specific Output
- **Corner-based Position Mapping**: Uses the 4 corners to map pieces to true squares
- **Perspective Correction**: Works for angled cameras and partial boards
- **Standardized JSON Output**: Returns human-readable square names

## Why YOLOv8

**Why YOLOv8?**
1. **Object Detection**: Perfect for detecting multiple chess pieces with bounding boxes
2. **Real-time Performance**: Fast inference suitable for applications
3. **Transfer Learning**: Pre-trained weights speed up training
4. **Easy Integration**: Excellent Python API and documentation
5. **Google Colab Support**: Works seamlessly with GPU acceleration

## Why Two Models

### The Core Problem
When YOLO detects a chess piece, it gives you pixel coordinates like "white rook at (450, 200)". But how do you convert that to chess notation like "a1"? 

The naive approach is to divide the image into an 8x8 grid and map pixels directly. This works if:
- The chessboard perfectly fills the image
- The camera is directly overhead  
- There's no perspective distortion

In reality, chess photos are taken from angles, boards don't fill the entire frame, and perspective makes squares look different sizes.

### The Solution: Two Models Working Together

**Model 1: Piece Detection** (YOLOv8m)
- Detects all chess pieces and their pixel locations
- Trained on 43,149 piece annotations

**Model 2: Corner Detection** (YOLOv8n) 
- Detects the 4 corners of the chessboard
- Trained on the same images with corner annotations
- Uses the corner positions to map the board geometry

### How They Work Together

1. **Corner model** finds the chessboard boundaries
2. **Piece model** finds all the pieces  
3. **Math** maps piece positions relative to the detected corners
4. **Result** accurate chess positions regardless of camera angle

### What we actually do (in simple terms)

- Detect the four board corners, then compute a perspective transform (a "homography") that "unwarps" the photo to a flat, top‑down 800×800 board.
- For each detected piece box, take a thin strip along the bottom of the box (where a piece touches its square), project that strip into the flat board, and pick the square it overlaps the most.
- If multiple detections land on the same square, keep the one with the highest confidence.

This is exactly the unwarping approach described above, and it’s what this repo implements end‑to‑end.

### Why This Approach Works

Instead of assuming the board fills the image, you now know exactly where the board is. You can handle:
- Tilted camera angles
- Boards that don't fill the frame
- Perspective distortion
- Different image sizes

The corner detection is much simpler than piece detection (just 4 points vs dozens of pieces), so it trains quickly and runs fast.

## Results (from my notebook)

- Pieces model (YOLOv8m): good detection coverage; some class confusion remains (queens vs bishops, etc.).
- Corners model (YOLOv8n): reliably detects board corners; enables accurate mapping to squares.
- Combined system: robust square assignment on angled photos; much better than grid-only.

(I trained with 100 epochs at 832 size for pieces and a lighter setup for corners. Exact scores depend on run, but the two-model setup clearly improved correct square placement.)

## Possible Improvements

1. Improve class accuracy (some pieces are visually similar at a distance)
2. Train a small classifier head per crop to refine piece type
3. Use temporal smoothing for video (tracks pieces over time)

## Limitations

- **Board orientation is unknown**: The system outputs consistent squares assuming a fixed orientation, but from a single image it does not know which side is a↔h or 1↔8. In other words, it recovers a correct layout of pieces relative to each other, but the absolute naming (e.g., whether the bottom-left is a1 or h8) can be ambiguous.
- **What to improve**: Try all four possible board orientations and score each one against a library of valid chess positions or heuristics (e.g., pawns start on ranks 2/7, kings/queens typical placement, legal piece counts). Pick the highest-scoring orientation. With a good prior or small position dataset, this should be highly accurate.

## Dependencies

- ultralytics (YOLOv8)
- opencv-python
- numpy
- matplotlib
- pyyaml
- torch (PyTorch)

## License and Dataset Credit

This project uses the Chess Recognition Dataset (ChessReD) by Athanasios Masouris. Please cite the original dataset if you use this work in research.

## Citation and Thanks

Thanks to Athanasios Masouris and Jan van Gemert. If you use this code or dataset in your research, please consider citing:

```bibtex
@conference{visapp24,
author={Athanasios Masouris. and Jan {van Gemert}.},
title={End-to-End Chess Recognition},
booktitle={Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
year={2024},
pages={393-403},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012370200003660},
isbn={978-989-758-679-8},
issn={2184-4321},
}
```



