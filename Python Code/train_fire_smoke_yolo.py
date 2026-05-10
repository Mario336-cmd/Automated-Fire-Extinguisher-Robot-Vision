from pathlib import Path
import argparse
import shutil


# Main project folder.
PROJECT_PATH = Path(r"C:\Users\uushu\Desktop\EME Fire Robot")

# YOLO data configuration file for the combined dataset.
DATA_YAML_PATH = PROJECT_PATH / "Datasets" / "Combined YOLO Fire Smoke Dataset" / "data.yaml"

# Default YOLO checkpoint used as the starting model.
DEFAULT_MODEL_PATH = PROJECT_PATH / "AI Model" / "yolo26s.pt"

# Folder where Ultralytics saves training and evaluation results.
RUNS_PATH = PROJECT_PATH / "AI Model" / "runs"

# Final model path used after training is complete.
FINAL_MODEL_PATH = PROJECT_PATH / "AI Model" / "fire_smoke_model.pt"

# Dataset splits expected by the training script.
DATASET_SPLITS = ["train", "val", "test"]

# Image file types accepted by the dataset check.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Counts files in a folder that match the allowed file extensions.
def count_files_with_extensions(folder_path: Path, allowed_extensions: set[str]) -> int:
    return sum(
        1
        for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
    )


# Checks that the dataset folders and label files are ready for YOLO training.
def validate_dataset() -> None:
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML_PATH}")

    # The combined dataset folder is the parent folder of data.yaml.
    dataset_path = DATA_YAML_PATH.parent

    print("Dataset check:")

    for dataset_split in DATASET_SPLITS:
        # Each split must have matching images and labels folders.
        images_folder_path = dataset_path / dataset_split / "images"
        labels_folder_path = dataset_path / dataset_split / "labels"

        if not images_folder_path.exists():
            raise FileNotFoundError(f"Images folder not found: {images_folder_path}")

        if not labels_folder_path.exists():
            raise FileNotFoundError(f"Labels folder not found: {labels_folder_path}")

        # Count the image files and YOLO label text files in this split.
        image_count = count_files_with_extensions(images_folder_path, IMAGE_EXTENSIONS)
        label_count = count_files_with_extensions(labels_folder_path, {".txt"})

        print(f"- {dataset_split}: {image_count} images, {label_count} labels")

        # Stop training if an image does not have a matching label file.
        if image_count != label_count:
            raise ValueError(
                f"{dataset_split} has {image_count} images but {label_count} labels."
            )


# Reads command-line settings so training can be changed without editing the script.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the fire/smoke dataset.")

    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO model/checkpoint path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=0, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--name", default="fire_smoke_yolo", help="Run name.")

    # Allows the dataset check to run without starting a full training job.
    parser.add_argument("--check-only", action="store_true", help="Only validate dataset and exit.")

    return parser.parse_args()


def main() -> None:
    # Get the training settings from the command line.
    args = parse_arguments()

    # Always validate the dataset before training starts.
    validate_dataset()

    # Stop here when only the dataset check was requested.
    if args.check_only:
        print("Check complete. Training was not started.")
        return

    model_path = Path(args.model)

    # Make sure the starting model/checkpoint exists before loading YOLO.
    if not model_path.exists():
        raise FileNotFoundError(f"Model/checkpoint not found: {model_path}")

    # Import YOLO after the checks so check-only mode does not need to load it.
    from ultralytics import YOLO

    # Load the starting YOLO model/checkpoint.
    model = YOLO(str(model_path))

    print("\nStarting training...")

    # Train the model and save the run outputs inside AI Model/runs.
    train_results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_PATH),
        name=args.name,
        patience=25,
        plots=True,
        save=True,
        val=True,
    )

    # The best model from the training run is saved in the run's weights folder.
    best_model_path = Path(train_results.save_dir) / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best trained model not found: {best_model_path}")

    # Copy the best model to a simple final path for robot or webcam testing.
    shutil.copy2(best_model_path, FINAL_MODEL_PATH)
    print(f"\nBest model copied to: {FINAL_MODEL_PATH}")

    print("\nRunning final test-set evaluation...")

    # Evaluate the trained model on the separate test split.
    model.val(
        data=str(DATA_YAML_PATH),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_PATH),
        name=f"{args.name}_test",
        plots=True,
    )

    print("\nTraining complete.")


# Run main only when this file is executed directly.
if __name__ == "__main__":
    main()
