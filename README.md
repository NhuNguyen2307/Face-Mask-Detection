# Face Mask Detection - DS 423 SA, Group 11

**Duy Tan University, Da Nang**
**Course:** DS 423 SA - Machine Learning with Large Datasets
**Interest area:** Object Detection

| Member | Student ID | Role |
|---|---:|---|
| Tran Van Truong | 28211452515 | Data & Analysis Lead |
| Nguyen Nhu Nguyen | 28211403542 | Modeling & Evaluation Lead |

## 1. Project Overview

This project builds an object-detection system for public-health mask-compliance monitoring. Given an image, the model locates each face with a bounding box and classifies it as one of three classes: `with_mask`, `without_mask`, or `mask_weared_incorrect`.

The final detector is **YOLOv8s (Ultralytics)**, initialized from COCO-pretrained weights and fine-tuned on the Kaggle Face Mask Detection dataset by andrewmvd. The project follows the Group 11 assignment requirements: annotations, training, detection, and evaluation with mAP, IoU, Precision, and Recall.

## 2. Code Availability

The complete, runnable, and commented notebook (`group11_facemask.ipynb`), along with this README, the LaTeX report source, and the trained weights, is hosted at:

- **GitHub repository:** [https://github.com/NhuNguyen2307/Face-Mask-Detection](https://github.com/NhuNguyen2307/Face-Mask-Detection.git)
- **Google Drive (dataset, training outputs, pretrained weights):** [https://drive.google.com/drive/folders/1d2VqVXUiyYbnAjqdw4f5thnWs3qOC1v-?usp=sharing](https://drive.google.com/drive/folders/1d2VqVXUiyYbnAjqdw4f5thnWs3qOC1v-?usp=sharing)

The GitHub repository contains the notebook, report source, and README. Large files (dataset, `runs/`, pretrained `.pt` weights) exceed GitHub's size limits and are hosted on Google Drive instead — see Section 4 for setup instructions.

## 3. Repository Structure

Actual project layout after running `group11_facemask.ipynb` (verified against the local working directory):

```text
.
|-- .gitignore                          # Excludes large/generated files from git
|-- group11_facemask.ipynb              # Main runnable notebook, Parts 1-10
|-- README.md                           # Project documentation
|-- report.tex                          # LaTeX report source, compile with XeLaTeX
|-- report.pdf                          # Compiled report
|-- yolov8s.pt                          # YOLOv8s pretrained weights, local cache
|-- yolov8n.pt                          # Extra local YOLO weight file, not used by final run
|
|-- data/
|   |-- face-mask-detection/            # Raw Kaggle dataset
|   |   |-- annotations/                # *.xml (Pascal-VOC annotations)
|   |   `-- images/                     # *.png
|   `-- face-mask-yolo/                 # Auto-generated in Part 6
|       |-- data.yaml                   # YOLO dataset config (paths + 3 classes)
|       |-- images/                     # train/, val/ subfolders (*.png)
|       `-- labels/                     # train/, val/ subfolders (*.txt)
|
|-- outputs/
|   |-- results_summary.csv             # Final metrics used in README/report
|   |-- charts/                         # EDA, loss, metric, and prediction figures
|   |   |-- ap_precision_recall_per_class.png
|   |   |-- eda_overview.png
|   |   |-- eda_sample_images.png
|   |   |-- feature_engineering.png
|   |   |-- loss_curve.png
|   |   `-- sample_predictions.png
|   |-- lr_search_001/                  # Learning-rate trial: lr0=0.01
|   |   |-- args.yaml
|   |   |-- results.csv
|   |   `-- weights/
|   |-- lr_search_0005/                 # Learning-rate trial: lr0=0.005
|   |   |-- args.yaml
|   |   |-- results.csv
|   |   `-- weights/
|   |-- lr_search_0001/                 # Learning-rate trial: lr0=0.001
|   |   |-- args.yaml
|   |   |-- results.csv
|   |   `-- weights/
|   |-- models/                         # Final copied model weights
|   |   |-- yolov8s_facemask_best.pt
|   |   `-- yolov8s_facemask_last.pt
|   `-- yolov8s_facemask_final/         # Ultralytics final training run (60 epochs)
|       |-- args.yaml
|       |-- results.csv
|       |-- results.png                 # Combined loss / metric curves
|       |-- confusion_matrix.png
|       |-- confusion_matrix_normalized.png
|       |-- F1_curve.png
|       |-- P_curve.png
|       |-- PR_curve.png
|       |-- R_curve.png
|       |-- labels.jpg
|       |-- labels_correlogram.jpg
|       |-- train_batch0.jpg / train_batch1.jpg / train_batch2.jpg
|       |-- train_batch4300.jpg / train_batch4301.jpg / train_batch4302.jpg
|       |-- val_batch0_labels.jpg / val_batch0_pred.jpg
|       |-- val_batch1_labels.jpg / val_batch1_pred.jpg
|       |-- val_batch2_labels.jpg / val_batch2_pred.jpg
|       `-- weights/                    # best.pt, last.pt
|
|-- report/
|   `-- figures/                        # Figures embedded in report.tex / report.pdf
|       |-- ap_precision_recall_per_class.png
|       |-- eda_overview.png
|       |-- loss_curve.png
|       `-- sample_predictions.png
|
`-- runs/
    `-- detect/
        `-- val/                        # Extra validation/prediction output from Ultralytics
```

## 4. Requirements

Recommended environment, matching the tested notebook:

- Windows 11 or another OS with Python 3.10+
- VS Code + Jupyter, JupyterLab, or classic Notebook
- NVIDIA CUDA GPU recommended; the run was tested on an NVIDIA GeForce RTX 3050 Laptop GPU with 4GB VRAM
- CPU execution is possible but training will be much slower

Install dependencies once:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python pandas numpy matplotlib seaborn scikit-learn pyyaml
pip install ultralytics
```

For CPU-only machines, install the CPU build of PyTorch instead:

```bash
pip install torch torchvision
```

## 5. Dataset Setup

Dataset: [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) by andrewmvd on Kaggle.

Expected raw dataset location:

```text
data/face-mask-detection/images/*.png
data/face-mask-detection/annotations/*.xml
```

**Manual download:**

1. Download the dataset ZIP from Kaggle.
2. Extract the `images/` and `annotations/` folders into `data/face-mask-detection/`.

**Kaggle API option:**

```bash
pip install kaggle
kaggle datasets download -d andrewmvd/face-mask-detection -p data --unzip
```

If the API extraction creates a different folder layout, move the extracted `images/` and `annotations/` folders into `data/face-mask-detection/` before running the notebook.

### Additional Resources (Google Drive)

Due to GitHub file-size limitations, the dataset, training outputs, and pretrained YOLO weights are hosted on Google Drive:

[https://drive.google.com/drive/folders/1d2VqVXUiyYbnAjqdw4f5thnWs3qOC1v-?usp=sharing](https://drive.google.com/drive/folders/1d2VqVXUiyYbnAjqdw4f5thnWs3qOC1v-?usp=sharing)

**Included resources:**

- Dataset (`data/`)
- Training outputs (`runs/`)
- Additional experiment files
- Pretrained weights (`yolov8n.pt`, `yolov8s.pt`)

**Setup steps after downloading from Google Drive:**

1. Place the `data/` folder in the project root directory.
2. Place `yolov8n.pt` and `yolov8s.pt` in the project root directory.
3. Open `group11_facemask.ipynb` and run all notebook cells in order.

## 6. How to Run the Notebook

Open `group11_facemask.ipynb` and run the cells from top to bottom. The notebook is organized into the 10 assigned project parts:

| Part | Notebook section | Owner |
|---:|---|---|
| 1 | Data collection and loading: parse Pascal-VOC XML annotations into a DataFrame | Tran Van Truong |
| 2 | Data cleaning and preprocessing: missing values, invalid labels, missing images, and out-of-bound box clamping | Tran Van Truong |
| 3 | Exploratory Data Analysis: class distribution, boxes per image, image sizes, and sample annotated images | Tran Van Truong |
| 4 | Feature engineering: box width, height, area, aspect ratio, and area ratio | Tran Van Truong |
| 5 | Stratified train/validation split for shared data preparation | Tran Van Truong |
| 6 | Model selection and building: convert cleaned annotations to YOLO format and initialize YOLOv8s | Nguyen Nhu Nguyen |
| 7 | Training setup, learning-rate search, and final training | Nguyen Nhu Nguyen |
| 8 | Evaluation with mAP, IoU, Precision, and Recall | Nguyen Nhu Nguyen |
| 9 | Prediction visualization on validation samples | Nguyen Nhu Nguyen |
| 10 | Save final model weights and `outputs/results_summary.csv` | Nguyen Nhu Nguyen |

The notebook creates the YOLO dataset under `data/face-mask-yolo/`, saves charts to `outputs/charts/`, saves model weights to `outputs/models/`, and writes the final metric table to `outputs/results_summary.csv`.

## 7. Training Configuration

| Setting | Value used in the completed run |
|---|---|
| Model | YOLOv8s, COCO-pretrained (`yolov8s.pt`) |
| Number of classes | 3 |
| Image size | 416 |
| Batch size | 8 |
| Seed | 42 |
| Device | CUDA device `0` when available; CPU fallback otherwise |
| DataLoader workers | 0, to avoid Windows/Jupyter multiprocessing issues |
| Mixed precision | Enabled (`amp=True`) |
| Learning-rate candidates | `0.01`, `0.005`, `0.001` |
| Best learning rate (`lr0`, from search) | `0.005` |
| Optimizer | Ultralytics `auto` optimizer (resolved to AdamW, lr≈0.001429), as recorded in `outputs/yolov8s_facemask_final/args.yaml` |
| Epochs | 60 |
| Early stopping patience | 15 |
| Final run directory | `outputs/yolov8s_facemask_final/` |

> **Note:** the notebook calls `model.train()` with `optimizer="auto"`, so Ultralytics automatically resolves its own optimizer/learning rate (AdamW, lr≈0.001429) and overrides the supplied `lr0=0.005`. The learning-rate search still correctly identified the best candidate among the three tested; see `report.tex` (Section 3.5) for the full discussion.

## 8. Results and Output Files

Final metrics from `outputs/results_summary.csv`:

| Metric | Value |
|---|---:|
| mAP@[0.5:0.95] | 0.5723 |
| mAP@0.50 | 0.8186 |
| mAP@0.75 | 0.6795 |
| Precision | 0.9293 |
| Recall | 0.7432 |

Per-class metrics (validation split, 171 images / 763 instances):

| Class | Instances | Precision | Recall | mAP@0.50 | mAP@[.5:.95] |
|---|---:|---:|---:|---:|---:|
| with_mask | 580 | 0.943 | 0.912 | 0.958 | 0.687 |
| without_mask | 156 | 0.902 | 0.705 | 0.800 | 0.523 |
| mask_weared_incorrect | 27 | 0.943 | 0.612 | 0.698 | 0.508 |

Main output files:

| Output | Path |
|---|---|
| Summary metrics | `outputs/results_summary.csv` |
| Best model weights | `outputs/models/yolov8s_facemask_best.pt` |
| Last model weights | `outputs/models/yolov8s_facemask_last.pt` |
| Training curves | `outputs/charts/loss_curve.png` |
| Per-class AP/Precision/Recall chart | `outputs/charts/ap_precision_recall_per_class.png` |
| Sample predictions | `outputs/charts/sample_predictions.png` |
| EDA overview | `outputs/charts/eda_overview.png` |
| LaTeX report source / compiled PDF | `report.tex` / `report.pdf` |
| LaTeX report figures | `report/figures/` |

## 9. Individual Contribution

| Member | Main contribution | Share |
|---|---|---:|
| Tran Van Truong | Data collection and loading, cleaning/preprocessing, EDA, feature engineering, and APA reference list | 50% |
| Nguyen Nhu Nguyen | Model selection, YOLOv8s training, hyperparameter tuning, evaluation, result charts, PowerPoint, LaTeX report, and conclusion | 50% |

Both members integrated and reviewed the final notebook before submission.

## 10. Limitations and Future Work

- The dataset is small for object detection: 853 images with three classes.
- The class distribution is imbalanced; `mask_weared_incorrect` has fewer examples and lower recall than `with_mask`.
- Training used `imgsz=416` and `batch=8` to fit the 4GB VRAM budget; a larger GPU could support higher image size or larger YOLO variants.
- Future work should add more incorrect-mask examples, evaluate on real CCTV/video streams, measure inference FPS, and export the model to ONNX or TensorRT for deployment.

## 11. References

andrewmvd. (2020). *Face Mask Detection* [Data set]. Kaggle. https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information, 11*(2), 125. https://doi.org/10.3390/info11020125

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.1.0) [Computer software]. https://github.com/ultralytics/ultralytics

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*, 8024-8035.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779-788. https://doi.org/10.1109/CVPR.2016.91

Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems, 28*, 91-99.
