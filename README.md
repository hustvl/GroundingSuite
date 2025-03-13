<div align="center">
<h1>GroundingSuite</h1>
<h3>Measuring Complex Multi-Granular Pixel Grounding</h3>

[Rui Hu](https://github.com/isfinne)<sup>1,\*</sup>, [Lianghui Zhu](https://scholar.google.com/citations?user=NvMHcs0AAAAJ&hl=zh-CN)<sup>1,\*</sup>, [Yuxuan Zhang](https://github.com/CoderZhangYx)<sup>1</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,🌟</sup>, Lei Liu<sup>2</sup>, Heng Liu<sup>2</sup>, Longjin Ran<sup>2</sup>,<br>Xiaoxin Chen<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> vivo AI Lab

(\* equal contribution, 🌟 Project lead, 📧 corresponding author)


[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<div align="center">
<img src="./assets/teaser.png">
</div>

## 📋 News

`[2025-3-14]:` GroundingSuite [arXiv]() paper released. 

## 🛠️ Pipeline

<div align="center">
<img src="./assets/pipeline.png">
</div>

## 📊 Dataset

GSEval uses images from the COCO dataset's unlabeled2017 split. You can download the dataset from: [Hugging Face](https://huggingface.co/datasets/hustvl/GSEval)

## 🚀 Usage

### Basic Usage

```bash
python evaluate_grounding.py --image_dir ./images --gt_file GroundingSuite-Eval.jsonl --pred_file model_predictions.jsonl
```

### Parameters

- `--image_dir`: Directory containing images (default: current directory)
- `--gt_file`: Path to ground truth JSONL file (default: "GroundingSuite-Eval.jsonl")
- `--pred_file`: Path to model prediction JSONL file (default: "claude_predictions.jsonl")
- `--output_file`: Path for saving evaluation results (default: "[model_name]_result.json")
- `--iou_threshold`: IoU threshold for evaluation (default: 0.5)
- `--vis_dir`: Directory for visualization results (default: "visualization")
- `--visualize`: Enable visualization generation (default: False)
- `--normalize_coords`: Whether prediction coordinates are normalized [0-1] (default: False)
- `--mode`: Evaluation mode ("box" or "mask") (default: "box")
- `--vis_samples`: Number of random samples to visualize (default: 5)

### Visualization Example

Generate visualizations comparing ground truth and predictions:

```bash
python evaluate_grounding.py --image_dir ./images --gt_file GroundingSuite-Eval.jsonl --pred_file model_predictions.jsonl --visualize --vis_dir ./vis_results
```

### Data Format

#### Ground Truth File Format (JSONL)

```json
{"idx": 1, "image_path": "images/example.jpg", "box": [10, 20, 100, 200], "class_id": 0, "label": "dog"}
```

#### Prediction File Format (JSONL)

```json
{"idx": 1, "image_path": "images/example.jpg", "box": [15, 25, 105, 205]}
```

### Evaluation Metrics

- **Box Mode**: Calculates IoU (Intersection over Union) and accuracy (IoU > threshold)
- **Mask Mode**: Calculates GIoU (mean IoU)

## 📚 Citation

If you find GroundingSuite useful in your research or applications, please consider giving us a star ⭐ and citing it using the following BibTeX entry:

```bibtex

```
