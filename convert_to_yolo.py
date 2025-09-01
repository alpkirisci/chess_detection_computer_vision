#!/usr/bin/env python3
"""
Convert ChessReD2K dataset to YOLOv8 format for training.
YOLOv8 format:
- Images in train/val/test folders
- Labels in corresponding txt files with format: class_id center_x center_y width height (normalized)
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import yaml

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [center_x, center_y, width, height] (normalized)
    """
    x, y, w, h = bbox
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return center_x, center_y, norm_w, norm_h

def create_yolo_dataset():
    print("Loading filtered ChessReD2K annotations...")
    with open('chessred2k_annotations.json', 'r') as f:
        data = json.load(f)
    
    # Create output directory structure
    output_dir = Path('yolo_dataset')
    output_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get splits information
    splits = data.get('splits', {})
    print(f"Splits found: {list(splits.keys())}")
    
    # If no splits available, create a simple train/val/test split
    if not splits:
        print("No splits found, creating 70/15/15 split...")
        images = data['images']
        total = len(images)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        splits = {
            'train': [img['id'] for img in images[:train_end]],
            'val': [img['id'] for img in images[train_end:val_end]],
            'test': [img['id'] for img in images[val_end:]]
        }
    
    # Create image_id to image mapping
    id_to_image = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    image_annotations = defaultdict(list)
    for ann in data['annotations']['pieces']:
        image_annotations[ann['image_id']].append(ann)
    
    # Process each split
    for split_name, image_ids in splits.items():
        print(f"Processing {split_name} split with {len(image_ids)} images...")
        
        for img_id in image_ids:
            if img_id not in id_to_image:
                continue
                
            img_info = id_to_image[img_id]
            img_path = Path('chessred2k') / img_info['path']
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Copy image
            dest_img_path = output_dir / split_name / 'images' / img_info['file_name']
            shutil.copy2(img_path, dest_img_path)
            
            # Create label file
            label_path = output_dir / split_name / 'labels' / (Path(img_info['file_name']).stem + '.txt')
            
            with open(label_path, 'w') as f:
                for ann in image_annotations[img_id]:
                    # Convert bbox to YOLO format
                    center_x, center_y, width, height = convert_bbox_to_yolo(
                        ann['bbox'], img_info['width'], img_info['height']
                    )
                    
                    # Write in YOLO format: class_id center_x center_y width height
                    f.write(f"{ann['category_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # Create class names mapping
    class_names = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create YOLO dataset configuration file
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),  # number of classes
        'names': [class_names[i] for i in sorted(class_names.keys())]
    }
    
    config_path = output_dir / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset conversion completed!")
    print(f"Output directory: {output_dir}")
    print(f"Configuration file: {config_path}")
    print(f"Classes: {dataset_config['names']}")
    
    # Print statistics
    for split_name in ['train', 'val', 'test']:
        img_count = len(list((output_dir / split_name / 'images').glob('*')))
        label_count = len(list((output_dir / split_name / 'labels').glob('*')))
        print(f"{split_name}: {img_count} images, {label_count} labels")

if __name__ == "__main__":
    create_yolo_dataset()
