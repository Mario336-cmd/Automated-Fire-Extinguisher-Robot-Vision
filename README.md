# Computer Vision System for Fire Extinguisher Robot

This project contains the computer vision part of an automated fire-extinguishing robot. The goal of the system is to help the robot detect possible fire-related danger from camera images using a trained YOLO object detection model.

Instead of relying only on a fire sensor, the robot can use the model output to identify the general location of fire or smoke in an image. The model detects one combined danger class called `fire_smoke` and returns bounding boxes around detected areas.

## Dataset Preparation

The computer vision work used six public object detection datasets with existing bounding-box annotations. These datasets were selected because YOLO needs both the image and the location of the object inside the image.

| Dataset ZIP file | Source |
| --- | --- |
| `D Fire Dataset Gaia Github.zip` | https://github.com/gaia-solutions-on-demand/DFireDataset |
| `Fire Dataset Pengbo0 Github.zip` | https://github.com/PengBo0/Home-fire-dataset |
| `Fire Smoke Detection Dataset Hussain Nasir Khan Kaggle.zip` | https://www.kaggle.com/datasets/hussainnasirkhan/fire-and-smoke-detection-dataset |
| `Fire Smoke Obstacle Dataset Roboflow.zip` | https://universe.roboflow.com/myworkspace-d5kq3/fire-smoke-obstacle-dataset/dataset/10 |
| `Fire Smoke YOLO Dataset Azimjaan21 Kaggle.zip` | https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo |
| `Indoor Fire Smoke Dataset Zenodo.zip` | https://zenodo.org/records/15826133 |

The ZIP files are stored in `Datasets/Original Zip Folders`. The datasets were combined into one YOLO dataset so the model could be trained from a single organized structure:

```text
Datasets/Combined YOLO Fire Smoke Dataset/
|-- train/
|   |-- images/
|   `-- labels/
|-- val/
|   |-- images/
|   `-- labels/
|-- test/
|   |-- images/
|   `-- labels/
`-- data.yaml
```

During combination, the images and labels were copied into the final dataset, renamed to avoid duplicate filenames, and checked for validity. The final dataset contained:

| Split | Images |
| --- | ---: |
| Training | 54,806 |
| Validation | 10,774 |
| Testing | 11,366 |
| Total | 76,946 |

The cleaning process checked 76,946 annotation files. A total of 31 invalid annotation rows were skipped, and no images were removed. The skipped rows were caused by out-of-range YOLO bounding-box values, such as zero width or height, negative values, or values greater than 1.

## Class Mapping

The original datasets used different class labels for fire and smoke. For this version of the robot, fire and smoke were combined into one danger class because the robot only needs to know that a possible fire-related hazard is present.

The final YOLO class setup is:

```yaml
nc: 1
names: ['fire_smoke']
```

All valid fire and smoke bounding boxes were mapped to class ID `0`.

## Training And Evaluation

The model was fine-tuned from the `yolo26s.pt` checkpoint using the combined fire and smoke dataset. It was not trained from scratch. The main training settings were:

| Setting | Value |
| --- | --- |
| Model | `yolo26s.pt` |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Device | 0 |
| Workers | 8 |
| Patience | 25 |

The Ultralytics YOLO library generated training losses, validation metrics, plots, and evaluation images during training. After training, the best model was copied to:

```text
AI Model/fire_smoke_model.pt
```

The trained model was evaluated on the separate test split, which was not used during training or validation. The test set contained 11,366 images and 16,298 `fire_smoke` instances.

| Metric | Result |
| --- | ---: |
| Precision | 0.755 |
| Recall | 0.657 |
| mAP50 | 0.737 |
| mAP50-95 | 0.421 |
| Best F1 score | 0.70 at 0.290 confidence |
| Inference speed | 4.1 ms per image |

The model correctly detected 11,372 `fire_smoke` objects, missed 4,926 objects, and produced 3,746 false positive detections. The normalized confusion matrix showed that 70% of true `fire_smoke` objects were detected and 30% were missed.

The best balance between precision and recall was reached at a confidence threshold of `0.290`. This can be used as a starting point for robot camera testing or webcam testing.

## Project Files

| Path | Purpose |
| --- | --- |
| `Computer Vision Documentation.pdf` | Main project documentation for the computer vision system. |
| `build_combined_dataset.py` | Builds the combined YOLO dataset from the original ZIP files. |
| `train_fire_smoke_yolo.py` | Validates the dataset, trains the YOLO model, copies the best model, and runs test evaluation. |
| `Datasets/Original Zip Folders` | Stores the six original dataset ZIP files. |
| `Datasets/Combined YOLO Fire Smoke Dataset` | Final combined YOLO dataset used for training, validation, and testing. |
| `AI Model/fire_smoke_model.pt` | Final trained fire and smoke detection model. |
| `AI Model/runs/fire_smoke_yolo_training_validation_metrics` | Training and validation metrics, plots, and sample predictions. |
| `AI Model/runs/fire_smoke_yolo_test_metrics` | Final test-set evaluation metrics, plots, and sample predictions. |

## Reproduction Notes

The scripts currently use hard-coded local Windows paths for this project folder. If the project is moved, update the path constants in the scripts before rebuilding the dataset or training the model.

To rebuild the combined dataset from the original ZIP files:

```powershell
python build_combined_dataset.py
```

Before running the script, `Datasets/Original Zip Folders` must already exist and contain the original dataset ZIP files. The script does not download the datasets.

This recreates `Datasets/Combined YOLO Fire Smoke Dataset`, so it should only be run when the dataset needs to be rebuilt. If that combined dataset folder already exists, the script deletes and rebuilds only that generated folder. It does not delete `Datasets/Original Zip Folders`, so the original ZIP files stay safe.

To check the dataset structure without starting training:

```powershell
python train_fire_smoke_yolo.py --check-only
```

To train with the default settings:

```powershell
python train_fire_smoke_yolo.py
```

## Limitations And Future Work

The model was trained using ready-made public datasets, so the images may not fully match the robot camera's actual view, lighting, distance, image quality, or angle. Fire and smoke were also combined into one `fire_smoke` class, which simplifies detection but does not allow the robot to treat fire and smoke separately.

Future work should include real-time testing with the robot camera, collecting a custom dataset from the robot's own environment, manually labeling new bounding boxes, and potentially separating fire and smoke into different classes for more detailed decision-making.
