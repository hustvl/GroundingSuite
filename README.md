<div align ="center">
<h1>GroundingSuite </h1>
<h3>Measuring Complex Multi-Granular Pixel Grounding</h3>

Rui Hu<sup>1,\*</sup>, [Lianghui Zhu](https://scholar.google.com/citations?user=NvMHcs0AAAAJ&hl=zh-CN)<sup>1,\*</sup>, [Yuxuan Zhang](https://github.com/CoderZhangYx)<sup>1</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,🌟</sup>, Lei Liu<sup>2</sup>, Heng Liu<sup>2</sup>, Longjin Ran<sup>2</sup>,<br>Xiaoxin Chen<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> vivo AI Lab

(\* equal contribution, 🌟 Project lead, 📧 corresponding author)


[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<div align="center">
<img src="./assets/teaser.png">
</div>

## 📋 News

`[2025-3-13]:` GroundingSuite [arXiv]() paper released. 





GroundingSuite is a toolkit for evaluating the grounding capabilities of vision-language models. It assesses how accurately models can locate objects in images, supporting both bounding box and segmentation mask evaluation modes.

## Dataset

This toolkit uses images from the COCO dataset's unlabeled2017 split. You can download the images from:
http://images.cocodataset.org/zips/unlabeled2017.zip

After downloading, extract the images to your preferred directory and specify this path using the `--image_dir` parameter when running the evaluation.

```bash
# Download COCO unlabeled images
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d ./images
```

## Usage

### Basic Usage

```bash
python evaluate_grounding.py --image_dir ./images --gt_file GroundingSuite-Eval.jsonl --pred_file model_predictions.jsonl
```

### Parameters

- `--image_dir`: Image directory path, defaults to current directory
- `--gt_file`: Ground truth JSONL file path, defaults to "GroundingSuite-Eval.jsonl"
- `--pred_file`: Model prediction JSONL file path, defaults to "claude_predictions.jsonl"
- `--output_file`: Output result file path, defaults to "[model_name]_result.json"
- `--iou_threshold`: IoU threshold, defaults to 0.5
- `--vis_dir`: Visualization results directory, defaults to "visualization"
- `--visualize`: Whether to generate visualization results, defaults to False
- `--normalize_coords`: Whether prediction coordinates are normalized (0-1), defaults to False
- `--mode`: Evaluation mode, can be "box" or "mask", defaults to "box"
- `--vis_samples`: Number of random samples to visualize, defaults to 5

### Visualization Example

To generate visualizations of ground truth and predictions:

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
If you find GroundingSuite useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.


```bibtex

```
