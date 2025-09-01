# Chess Piece Detection Project

This project detects chess pieces on real photos of boards and returns their positions in **algebraic notation** (like `e4`) in JSON.  
It uses **two YOLOv8 models** (pieces + corners) combined so it works even when the board is tilted, partially visible, or not centered.

<p align="center">
  <img width="100%" alt="piece_corner_detection_results" src="https://github.com/user-attachments/assets/beea985f-0e82-44a3-8047-8cd56fdb83e8" />
</p>
<p align="center"><em>Example of combined piece + corner detection</em></p>

---

## Project Overview

- **Input**: Images of chess boards during games  
- **Output**: JSON format with piece positions in algebraic notation  
- **Method**: YOLOv8 object detection  
- **Dataset**: ChessReD2K (2,078 images, 43,149 piece annotations)

### Example Output
```json
{
  "board_state": [
    { "position": "a1", "piece": "white_rook" },
    { "position": "g8", "piece": "black_knight" }
  ]
}
```

---

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
├── chess_inference_example.py    # Inference with robust corner-based mapping
└── chess_detection_training.ipynb # Google Colab training notebook
```

---

## Datasets (YOLO format)

- **Pieces YOLO dataset**: [Google Drive link](https://drive.google.com/drive/folders/1IMkCi-F_oXUQOYbPu6CLmcugU-qixrHH?usp=sharing)  
- **Corners YOLO dataset**: [Google Drive link](https://drive.google.com/drive/folders/1ZfEUz5iiaLxsD8v_erVTvj8yadNA_YcH?usp=sharing)

### ChessReD2K Subset
- **Total Images**: 2,078  
- **Piece Annotations**: 43,149  
- **Corner Annotations**: 2,078  
- **Game IDs**: 0, 6, 19, 22, 28, 33, 38, 41, 42, 47, 56, 58, 61, 72, 76, 78, 83, 87, 91, 99  

### Classes (13 total)
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
- Train: 1,454 images (70%)  
- Validation: 312 images (15%)  
- Test: 312 images (15%)  

---

## Method

### Why Two Models?

If you only detect pieces and split the image into an 8×8 grid, it fails when:
- The board is tilted  
- The camera is at an angle  
- The board doesn’t fill the frame  

**Solution** → Use two YOLOv8 models:
- **Pieces model (YOLOv8m)** → detects all chess pieces  
- **Corners model (YOLOv8n)** → detects 4 board corners  

### How Mapping Works

1. **Corner model** finds the board boundaries  
2. **Perspective transform (homography)** unwarps the board to a flat 800×800 grid  
3. **Pieces model** detects bounding boxes  
4. Take a thin strip at the bottom of each piece box → project into the rectified board  
5. Assign to the overlapping square  

<p align="center">
  <img width="45%" alt="rectified_board" src="https://github.com/user-attachments/assets/be9ccdf2-e55d-457e-8d25-14de41b82aad" />
  <img width="45%" alt="grid_analysis" src="https://github.com/user-attachments/assets/9e4f8b54-f40b-4da1-956e-9cce80fc7a09" />
</p>
<p align="center"><em>Left: rectified board after perspective transform | Right: grid overlay analysis</em></p>

---

## Results

- **Pieces model (YOLOv8m)** → good detection, some confusion (queen vs bishop)  
- **Corners model (YOLOv8n)** → reliable 4-corner detection  
- **Combined system** → robust square assignment on angled photos  

---

## Possible Improvements

1. Better class accuracy (queens vs bishops)  
2. Train a small classifier on crops to refine piece type  
3. Add temporal smoothing for video streams  

---

## Limitations

- **Board orientation is ambiguous**: from one image, we don’t know if bottom-left is `a1` or `h8`.  
- **Fix**: Try all 4 orientations and select the most plausible one (e.g., using valid chess position heuristics).  

---

## Dependencies

- ultralytics (YOLOv8)  
- opencv-python  
- numpy  
- matplotlib  
- pyyaml  
- torch (PyTorch)  

---

## License and Dataset Credit

This project uses the Chess Recognition Dataset (ChessReD) by **Athanasios Masouris**.  
Please cite the original dataset if you use this work in research.  

---

## Citation

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
