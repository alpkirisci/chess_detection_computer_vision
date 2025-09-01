#!/usr/bin/env python3
"""
Script to create corner detection dataset directly in Google Colab.
Run this in Colab after uploading chessred2k_annotations.json to your Drive.
"""

import json
import os
import shutil
from pathlib import Path
import yaml
from collections import defaultdict

def create_corner_yolo_dataset(yolo_dataset_path, annotations_file, output_dir):
    """
    Create YOLO corner detection dataset from existing yolo_dataset and annotations.
    
    Args:
        yolo_dataset_path: Path to existing yolo_dataset folder
        annotations_file: Path to chessred2k_annotations.json
        output_dir: Output directory for corner dataset
    """
    print("Creating corner detection dataset...")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create mapping from image_id to corner data
    corner_annotations = {}
    for corner_ann in data.get('annotations', {}).get('corners', []):
        corner_annotations[corner_ann['image_id']] = corner_ann['corners']
    
    # Create mapping from filename to image info
    filename_to_image = {}
    for img in data['images']:
        filename_to_image[img['file_name']] = img
    
    print(f"Found {len(corner_annotations)} corner annotations")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        source_images_dir = Path(yolo_dataset_path) / split / 'images'
        dest_images_dir = output_path / split / 'images'
        dest_labels_dir = output_path / split / 'labels'
        
        if not source_images_dir.exists():
            print(f"Warning: {source_images_dir} not found")
            continue
        
        image_files = list(source_images_dir.glob('*.jpg'))
        copied_count = 0
        
        print(f"Processing {split} split with {len(image_files)} images...")
        
        for image_file in image_files:
            filename = image_file.name
            
            # Get image info
            if filename not in filename_to_image:
                continue
            
            img_info = filename_to_image[filename]
            img_id = img_info['id']
            
            # Check if we have corner data for this image
            if img_id not in corner_annotations:
                continue
            
            corners = corner_annotations[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            dest_image_path = dest_images_dir / filename
            shutil.copy2(image_file, dest_image_path)
            
            # Create label file with corner points
            label_file = dest_labels_dir / (image_file.stem + '.txt')
            
            with open(label_file, 'w') as f:
                # Convert each corner to YOLO format (class_id x y width height)
                # We'll use point detection format: class_id center_x center_y small_width small_height
                
                corners_list = [
                    (corners['top_left'], 0),      # class 0: top_left (a8)
                    (corners['top_right'], 1),     # class 1: top_right (h8)
                    (corners['bottom_left'], 2),   # class 2: bottom_left (a1)
                    (corners['bottom_right'], 3),  # class 3: bottom_right (h1)
                ]
                
                for (x, y), class_id in corners_list:
                    # Normalize coordinates
                    norm_x = x / img_width
                    norm_y = y / img_height
                    
                    # Use small bounding box for point detection
                    box_size = 0.02  # 2% of image size
                    
                    f.write(f"{class_id} {norm_x:.6f} {norm_y:.6f} {box_size:.6f} {box_size:.6f}\n")
            
            copied_count += 1
        
        print(f"{split}: {copied_count} images with corner annotations")
    
    # Create YOLO dataset configuration for corners
    corner_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': 4,  # 4 corner classes
        'names': ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    }
    
    config_path = output_path / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(corner_config, f, default_flow_style=False)
    
    print(f"Corner dataset created at: {output_path}")
    print(f"Configuration: {config_path}")
    print(f"Classes: {corner_config['names']}")
    
    # Print final statistics
    for split in ['train', 'val', 'test']:
        img_count = len(list((output_path / split / 'images').glob('*')))
        label_count = len(list((output_path / split / 'labels').glob('*')))
        print(f"{split}: {img_count} images, {label_count} labels")
    
    return str(output_path)

if __name__ == "__main__":
    # For Google Colab usage
    print("Chess Corner Detection Dataset Creator")
    print("====================================")
    
    # Paths in Google Colab
    YOLO_DATASET_PATH = "/content/drive/MyDrive/yolo_dataset"
    ANNOTATIONS_FILE = "/content/drive/MyDrive/chessred2k_annotations.json"
    OUTPUT_DIR = "/content/drive/MyDrive/yolo_corner_dataset"
    
    # Check if paths exist
    if not os.path.exists(YOLO_DATASET_PATH):
        print(f"Error: {YOLO_DATASET_PATH} not found")
        print("Please upload yolo_dataset to your Google Drive")
        exit(1)
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: {ANNOTATIONS_FILE} not found") 
        print("Please upload chessred2k_annotations.json to your Google Drive")
        exit(1)
    
    print(f"Source dataset: {YOLO_DATASET_PATH}")
    print(f"Annotations: {ANNOTATIONS_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create corner dataset
    dataset_path = create_corner_yolo_dataset(
        YOLO_DATASET_PATH, 
        ANNOTATIONS_FILE, 
        OUTPUT_DIR
    )
    
    print(f"\n✅ Corner dataset created successfully!")
    print(f"📁 Dataset location: {dataset_path}")
    print(f"🚀 Ready for corner detection training!")
