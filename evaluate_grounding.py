import json
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorama
from colorama import Fore, Back, Style
from collections import defaultdict
from tqdm import tqdm

# Initialize colorama
colorama.init(autoreset=True)

class GroundingEvaluator:
    def __init__(self, image_dir: str = ".", gt_file: str = "GroundingSuite-Eval.jsonl", 
                 iou_threshold: float = 0.5, vis_dir: str = "visualization", 
                 visualize: bool = False, normalize_coords: bool = False,
                 mode: str = "box"):
        """Initialize Grounding Evaluator
        
        Args:
            image_dir: Image directory, default is "images"
            gt_file: Ground truth JSONL file path, default is "ground_truth.jsonl"
            iou_threshold: IoU threshold, default is 0.5
            vis_dir: Visualization results directory, default is "visualization"
            visualize: Whether to generate visualization results, default is False
            normalize_coords: Whether prediction coordinates are normalized (0-1), default is False
            mode: Evaluation mode, can be "box" or "mask", default is "box"
        """
        self.image_dir = image_dir
        self.gt_file = gt_file
        self.iou_threshold = iou_threshold
        self.vis_dir = vis_dir
        self.visualize = visualize
        self.normalize_coords = normalize_coords
        self.mode = mode
        self.gt_data = None
        
        # Validate mode
        if self.mode not in ["box", "mask"]:
            print(f"{Fore.RED}Warning: Invalid mode '{self.mode}', will use default mode 'box'")
            self.mode = "box"
        
        # Create visualization directory
        if self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        # Load ground truth data
        if os.path.exists(self.gt_file):
            self.gt_data = self.read_jsonl(self.gt_file)
            print(f"{Fore.GREEN}Loaded ground truth data: {len(self.gt_data)} items")
        else:
            print(f"{Fore.RED}Warning: Ground truth file {self.gt_file} does not exist")
    
    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Read JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes
        
        box format: [x_min, y_min, x_max, y_max]
        """
        # Calculate intersection area
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])
        
        # If no intersection, return 0
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
    
    def is_normalized_coordinates(self, coords: List[float]) -> bool:
        """Check if coordinates are normalized (within 0-1 range)"""
        return all(0 <= coord <= 1 for coord in coords)
    
    def convert_normalized_to_absolute(self, coords: List[float], image_path: str) -> List[float]:
        """Convert normalized coordinates to absolute coordinates"""
        img = Image.open(image_path)
        width, height = img.size
        
        absolute_coords = [
            coords[0] * width,
            coords[1] * height,
            coords[2] * width,
            coords[3] * height
        ]
        
        return absolute_coords
    
    def visualize_boxes(self, image_path: str, gt_box: List[float], pred_box: List[float], 
                       output_path: str, iou: float = None):
        """Visualize ground truth and predicted bounding boxes"""
        # Load image
        img = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        # Draw ground truth box (green)
        rect_gt = patches.Rectangle(
            (gt_box[0], gt_box[1]), 
            gt_box[2] - gt_box[0], 
            gt_box[3] - gt_box[1], 
            linewidth=2, 
            edgecolor='g', 
            facecolor='none',
            label='Ground Truth'
        )
        ax.add_patch(rect_gt)
        
        # Draw predicted box (red)
        if pred_box:
            rect_pred = patches.Rectangle(
                (pred_box[0], pred_box[1]), 
                pred_box[2] - pred_box[0], 
                pred_box[3] - pred_box[1], 
                linewidth=2, 
                edgecolor='r', 
                facecolor='none',
                label='Prediction'
            )
            ax.add_patch(rect_pred)
        
        # Add IoU information
        if iou is not None:
            plt.title(f'IoU: {iou:.4f}')
        
        plt.legend()
        
        # Save image
        plt.savefig(output_path)
        plt.close()
    
    def visualize_random_gt(self, num_samples: int = 5):
        """Randomly visualize ground truth data
        
        Args:
            num_samples: Number of samples to visualize, default is 5
        """
        if not self.gt_data:
            print(f"{Fore.RED}Error: Ground truth data not loaded")
            return
            
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir, exist_ok=True)
            
        # Randomly select samples
        sample_indices = np.random.choice(len(self.gt_data), min(num_samples, len(self.gt_data)), replace=False)
        
        print(f"{Fore.CYAN}Randomly visualizing {len(sample_indices)} ground truth samples:")
        
        for i, idx in enumerate(sample_indices):
            gt_item = self.gt_data[idx]
            image_path_rel = gt_item["image_path"]
            image_path = os.path.join(self.image_dir, image_path_rel)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"{Fore.RED}Warning: Image does not exist {image_path}")
                continue
                
            class_id = gt_item.get("class_id", 0)
            gt_idx = gt_item.get("idx")
            
            # Print current processing image
            print(f"{Fore.CYAN}Sample {i+1}/{len(sample_indices)}: {image_path_rel} (Class ID: {class_id}, IDX: {gt_idx})")
            
            # Visualization filename
            vis_filename = f"gt_random_{i}_{os.path.basename(image_path_rel).split('.')[0]}"
            if gt_idx is not None:
                vis_filename += f"_idx{gt_idx}"
            vis_filename += f"_class{class_id}_vis.jpg"
            vis_path = os.path.join(self.vis_dir, vis_filename)
            
            if self.mode == "box":
                gt_box = gt_item["box"]
                print(f"{Fore.BLUE}Ground truth box: {gt_box}")
                # Call visualization function, only pass ground truth box
                self.visualize_boxes(image_path, gt_box, None, vis_path)
            else:  # mask mode
                segmentation = gt_item.get("segmentation")
                if segmentation:
                    img = Image.open(image_path)
                    height, width = img.size[::-1]  # Note: need to reverse, as PIL and numpy have different dimension orders
                    gt_mask = self.rle_to_mask(segmentation, (height, width))
                    print(f"{Fore.BLUE}Ground truth mask: RLE format")
                    # Call visualization function, only pass ground truth mask
                    self.visualize_masks(image_path, gt_mask, None, vis_path)
                else:
                    print(f"{Fore.RED}Warning: No segmentation mask data found")
                    continue
            
            print(f"{Fore.GREEN}Visualization saved to: {vis_path}")
    
    def rle_to_mask(self, rle: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
        """Convert RLE format to binary mask
        
        Args:
            rle: RLE format mask, containing 'counts' and 'size' fields
            shape: Output mask shape (height, width)
            
        Returns:
            Binary mask with shape
        """
        if not rle:
            return None
            
        height, width = shape
        if 'counts' in rle and 'size' in rle:
            try:
                from pycocotools import mask as mask_utils
                mask = mask_utils.decode(rle)
                return mask
            except ImportError:
                print(f"{Fore.RED}Warning: pycocotools not installed, cannot decode RLE mask")
                return None
            except Exception as e:
                print(f"{Fore.RED}Warning: Error decoding RLE mask: {e}")
                return None
        return None
        
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two masks
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            IoU value
        """
        if mask1 is None or mask2 is None:
            return 0.0
            
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
        
    def visualize_masks(self, image_path: str, gt_mask: np.ndarray, pred_mask: np.ndarray, 
                       output_path: str, iou: float = None):
        """Visualize ground truth and predicted masks
        
        Args:
            image_path: Image path
            gt_mask: Ground truth mask
            pred_mask: Predicted mask
            output_path: Output image path
            iou: IoU value, optional
        """
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img_array)
        
        # Create mask overlays
        if gt_mask is not None:
            gt_mask_rgba = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4), dtype=np.uint8)
            gt_mask_rgba[gt_mask > 0] = [0, 255, 0, 128]  # Green, semi-transparent
            ax.imshow(gt_mask_rgba, alpha=0.5)
        
        if pred_mask is not None:
            pred_mask_rgba = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4), dtype=np.uint8)
            pred_mask_rgba[pred_mask > 0] = [255, 0, 0, 128]  # Red, semi-transparent
            ax.imshow(pred_mask_rgba, alpha=0.5)
        
        # Add legend
        gt_patch = patches.Patch(color='green', alpha=0.5, label='Ground Truth')
        pred_patch = patches.Patch(color='red', alpha=0.5, label='Prediction')
        ax.legend(handles=[gt_patch, pred_patch])
        
        # Add IoU information
        if iou is not None:
            plt.title(f'IoU: {iou:.4f}')
        
        # Save image
        plt.savefig(output_path)
        plt.close()
    
    def evaluate(self, pred_file: str, output_file: str = None):
        """Evaluate model's grounding capability
        
        Args:
            pred_file: Model prediction JSONL file path
            output_file: Output result file
        """
        # Check if ground truth data is loaded
        if not self.gt_data:
            print(f"{Fore.RED}Error: Ground truth data not loaded")
            return None, None, None
            
        # Read prediction data
        pred_data = self.read_jsonl(pred_file)
        
        # Check if data lengths match
        if len(self.gt_data) != len(pred_data):
            print(f"{Fore.RED}Warning: Ground truth data ({len(self.gt_data)} items) and prediction data ({len(pred_data)} items) lengths do not match")
        
        # Create mapping from idx to prediction data
        pred_map = {}
        for item in pred_data:
            idx = item.get("idx")
            if idx is not None:
                pred_map[idx] = item
        
        # If no idx field, use image path as alternative mapping
        if not pred_map:
            for item in pred_data:
                image_path = item.get("image_path")
                if image_path:
                    pred_map[image_path] = item
        
        results = []
        
        # Statistics by class_id
        class_counts = defaultdict(int)  # Total samples for each class_id
        class_correct_counts = defaultdict(int)  # Correct samples for each class_id
        class_ious = defaultdict(list)  # IoU values for each class_id (for mask mode)
        
        model_name = os.path.basename(pred_file).split('.')[0]
        print(f"{Fore.CYAN}Evaluating model: {model_name}")
        
        for idx, gt_item in enumerate(tqdm(self.gt_data)):
            image_path_rel = gt_item["image_path"]
            image_path = os.path.join(self.image_dir, image_path_rel)
            
            # Get class_id
            class_id = gt_item.get("class_id", 0)
            class_counts[class_id] += 1
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"{Fore.RED}Warning: Image does not exist {image_path}")
                continue
            
            # First try to match prediction data using idx
            gt_idx = gt_item.get("idx")
            pred_item = None
            if gt_idx is not None:
                pred_item = pred_map.get(gt_idx)
                if pred_item:
                    print(f"{Fore.GREEN}Matched prediction data using idx {gt_idx}")
            
            # If not matched by idx, try to match using image path
            if not pred_item:
                pred_item = pred_map.get(image_path_rel)
                if not pred_item:
                    print(f"{Fore.RED}Warning: No prediction data found for image {image_path_rel} or idx {gt_idx}")
                    continue
            
            # Print current processing image
            print(f"\n{Fore.CYAN}Processing image {idx+1}/{len(self.gt_data)}: {image_path_rel} (Class ID: {class_id}, IDX: {gt_idx})")
            
            result = {
                "idx": gt_idx,
                "image_path": image_path_rel,
                "class_id": class_id,
                "label": gt_item.get("label", ""),
                "caption": gt_item.get("caption", ""),
                "correct": False
            }
            
            if self.mode == "box":
                # Get ground truth box
                gt_box = gt_item["box"]
                result["gt_box"] = gt_box
                print(f"{Fore.BLUE}Ground truth box: {gt_box}")
                
                # Get predicted box
                pred_box = pred_item.get("box") or pred_item.get("predicted_box")
                result["predicted_box"] = pred_box
                print(f"{Fore.MAGENTA}Predicted box: {pred_box}")
                
                # If prediction coordinates are normalized, convert to absolute
                if pred_box and self.normalize_coords and self.is_normalized_coordinates(pred_box):
                    pred_box = self.convert_normalized_to_absolute(pred_box, image_path)
                    print(f"{Fore.YELLOW}Converted normalized coordinates to absolute: {pred_box}")
                
                # Calculate IoU and determine if correct
                if pred_box:
                    iou = self.calculate_iou(gt_box, pred_box)
                    result["iou"] = iou
                    result["correct"] = iou >= self.iou_threshold
                    
                    if result["correct"]:
                        print(f"{Fore.GREEN}IoU: {iou:.4f}, Correct: {result['correct']}")
                        class_correct_counts[class_id] += 1
                    else:
                        print(f"{Fore.RED}IoU: {iou:.4f}, Correct: {result['correct']}")
                    
                    # Visualize boxes
                    if self.visualize and self.vis_dir:
                        vis_filename = f"{os.path.basename(image_path_rel).split('.')[0]}"
                        if gt_idx is not None:
                            vis_filename += f"_idx{gt_idx}"
                        vis_filename += f"_{model_name}_class{class_id}_vis.jpg"
                        vis_path = os.path.join(self.vis_dir, vis_filename)
                        self.visualize_boxes(image_path, gt_box, pred_box, vis_path, iou)
                        print(f"{Fore.CYAN}Visualization saved to: {vis_path}")
                else:
                    print(f"{Fore.RED}No predicted box found")
            else:  # mask mode
                # Get ground truth mask
                gt_segmentation = gt_item.get("segmentation")
                if not gt_segmentation:
                    print(f"{Fore.RED}Warning: No ground truth segmentation mask data found")
                    continue
                
                # Get predicted mask
                pred_segmentation = pred_item.get("segmentation") or pred_item.get("predicted_segmentation")
                if not pred_segmentation:
                    print(f"{Fore.RED}Warning: No predicted segmentation mask data found")
                    continue
                
                # Convert RLE to mask
                img = Image.open(image_path)
                height, width = img.size[::-1]  # Note: need to reverse
                gt_mask = self.rle_to_mask(gt_segmentation, (height, width))
                pred_mask = self.rle_to_mask(pred_segmentation, (height, width))
                
                result["gt_segmentation"] = gt_segmentation
                result["predicted_segmentation"] = pred_segmentation
                
                print(f"{Fore.BLUE}Ground truth mask: RLE format")
                print(f"{Fore.MAGENTA}Predicted mask: RLE format")
                
                # Calculate mask IoU
                if gt_mask is not None and pred_mask is not None:
                    iou = self.calculate_mask_iou(gt_mask, pred_mask)
                    result["iou"] = iou
                    # For mask mode, we don't use threshold-based correctness
                    # Instead, we collect all IoU values for GIoU calculation
                    class_ious[class_id].append(iou)
                    
                    print(f"{Fore.BLUE}Mask IoU: {iou:.4f}")
                    
                    # Visualize masks
                    if self.visualize and self.vis_dir:
                        vis_filename = f"{os.path.basename(image_path_rel).split('.')[0]}"
                        if gt_idx is not None:
                            vis_filename += f"_idx{gt_idx}"
                        vis_filename += f"_{model_name}_mask_class{class_id}_vis.jpg"
                        vis_path = os.path.join(self.vis_dir, vis_filename)
                        self.visualize_masks(image_path, gt_mask, pred_mask, vis_path, iou)
                        print(f"{Fore.CYAN}Mask visualization saved to: {vis_path}")
                else:
                    print(f"{Fore.RED}Cannot decode RLE masks")
            
            results.append(result)
        
        if self.mode == "box":
            # Calculate overall accuracy for box mode
            total_correct = sum(class_correct_counts.values())
            total_samples = len(results)
            overall_accuracy = total_correct / total_samples if total_samples else 0
            
            # Calculate accuracy for each class_id
            class_accuracies = {}
            for class_id in sorted(class_counts.keys()):
                if class_counts[class_id] > 0:
                    class_accuracies[class_id] = class_correct_counts[class_id] / class_counts[class_id]
                else:
                    class_accuracies[class_id] = 0.0
            
            # Print results
            print(f"\n{Fore.CYAN}{model_name} Summary Results:")
            print(f"{Fore.YELLOW}Overall Accuracy@{self.iou_threshold}: {overall_accuracy:.4f}")
            
            print(f"\n{Fore.CYAN}Results by Class ID:")
            for class_id in sorted(class_accuracies.keys()):
                correct = class_correct_counts[class_id]
                total = class_counts[class_id]
                acc = class_accuracies[class_id]
                print(f"{Fore.YELLOW}Class ID {class_id}: Accuracy@{self.iou_threshold} = {acc:.4f} ({correct}/{total})")
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model": model_name,
                        "mode": "box",
                        "overall_accuracy": overall_accuracy,
                        "iou_threshold": self.iou_threshold,
                        "class_accuracies": {str(k): v for k, v in class_accuracies.items()},
                        "class_counts": {str(k): v for k, v in class_counts.items()},
                        "class_correct_counts": {str(k): v for k, v in class_correct_counts.items()},
                        "results": results
                    }, f, ensure_ascii=False, indent=2)
                print(f"{Fore.GREEN}Detailed results saved to: {output_file}")
            
            return overall_accuracy, class_accuracies, results
        else:  # mask mode
            # Calculate GIoU (mean IoU) for mask mode
            class_gious = {}
            for class_id in sorted(class_ious.keys()):
                if class_ious[class_id]:
                    class_gious[class_id] = sum(class_ious[class_id]) / len(class_ious[class_id])
                else:
                    class_gious[class_id] = 0.0
            
            # Calculate overall GIoU
            all_ious = [iou for ious in class_ious.values() for iou in ious]
            overall_giou = sum(all_ious) / len(all_ious) if all_ious else 0
            
            # Print results
            print(f"\n{Fore.CYAN}{model_name} Summary Results:")
            print(f"{Fore.YELLOW}Overall GIoU (mean IoU): {overall_giou:.4f}")
            
            print(f"\n{Fore.CYAN}Results by Class ID:")
            for class_id in sorted(class_gious.keys()):
                count = len(class_ious[class_id])
                giou = class_gious[class_id]
                print(f"{Fore.YELLOW}Class ID {class_id}: GIoU = {giou:.4f} (samples: {count})")
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model": model_name,
                        "mode": "mask",
                        "overall_giou": overall_giou,
                        "class_gious": {str(k): v for k, v in class_gious.items()},
                        "class_counts": {str(k): len(v) for k, v in class_ious.items()},
                        "results": results
                    }, f, ensure_ascii=False, indent=2)
                print(f"{Fore.GREEN}Detailed results saved to: {output_file}")
            
            return overall_giou, class_gious, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s Grounding capability')
    parser.add_argument('--image_dir', type=str, default='.', help='Image directory')
    parser.add_argument('--gt_file', type=str, default='GroundingSuite-Eval.jsonl', help='Ground truth JSONL file path')
    parser.add_argument('--pred_file', type=str, default='claude_predictions.jsonl', help='Model prediction JSONL file path')
    parser.add_argument('--output_file', type=str, default=None, help='Output result file')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--vis_dir', type=str, default='visualization', help='Visualization results directory')
    parser.add_argument('--visualize', action='store_true', help='Whether to generate visualization results')
    parser.add_argument('--normalize_coords', action='store_true', help='Whether prediction coordinates are normalized (0-1)')
    parser.add_argument('--mode', type=str, choices=['box', 'mask'], default='box', help='Evaluation mode, can be "box" or "mask"')
    parser.add_argument('--vis_samples', type=int, default=5, help='Number of random samples to visualize')
    
    args = parser.parse_args()
    
    # Create evaluator instance
    evaluator = GroundingEvaluator(
        image_dir=args.image_dir,
        gt_file=args.gt_file,
        iou_threshold=args.iou_threshold,
        vis_dir=args.vis_dir,
        visualize=args.visualize,
        normalize_coords=args.normalize_coords,
        mode=args.mode
    )
    
    # Randomly visualize ground truth data
    if args.vis_samples > 0:
        evaluator.visualize_random_gt(num_samples=args.vis_samples)
    
    # Evaluate model
    if args.pred_file:
        output_file = args.output_file or f"{os.path.basename(args.pred_file).split('.')[0]}_result.json"
        evaluator.evaluate(
            pred_file=args.pred_file,
            output_file=output_file
        )

if __name__ == "__main__":
    main()
