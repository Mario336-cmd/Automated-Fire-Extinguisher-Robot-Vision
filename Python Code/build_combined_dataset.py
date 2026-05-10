from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO
import shutil

from PIL import Image, UnidentifiedImageError


# Main project folder.
PROJECT_PATH = Path(r"C:\Users\uushu\Desktop\EME Fire Robot")

# Folder containing the original dataset ZIP files.
ORIGINAL_ZIP_FOLDER_PATH = PROJECT_PATH / "Datasets" / "Original Zip Folders"

# Final combined dataset folder.
COMBINED_DATASET_PATH = PROJECT_PATH / "Datasets" / "Combined YOLO Fire Smoke Dataset"

# Dataset splits used by YOLO.
DATASET_SPLITS = ["train", "val", "test"]

# Image file types accepted by the script.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Final class name used in data.yaml.
FINAL_CLASS_NAME = "fire_smoke"


# Converts valid to val so all datasets use the same split name.
def normalize_split_name(split_name: str) -> str:
    if split_name == "valid":
        return "val"

    return split_name


# Skips unwanted hidden Mac files inside ZIP files.
def should_skip_zip_entry(zip_entry_path: PurePosixPath) -> bool:
    path_parts_lower = [path_part.lower() for path_part in zip_entry_path.parts]

    return "__macosx" in path_parts_lower or zip_entry_path.name.startswith("._")


# Finds whether an image belongs to train, val, or test.
def find_split_and_images_index(zip_entry_path: PurePosixPath) -> tuple[str, int] | None:
    path_parts_lower = [path_part.lower() for path_part in zip_entry_path.parts]

    if "images" not in path_parts_lower:
        return None

    images_index = path_parts_lower.index("images")

    if images_index == 0:
        return None

    split_name = normalize_split_name(path_parts_lower[images_index - 1])

    if split_name not in DATASET_SPLITS:
        return None

    return split_name, images_index


# Gets the matching YOLO label path for an image path.
def get_matching_label_name(image_name: str, images_index: int) -> str:
    image_path = PurePosixPath(image_name)
    label_path_parts = list(image_path.parts)

    # Replace the images folder with the labels folder.
    label_path_parts[images_index] = "labels"

    label_path = PurePosixPath(*label_path_parts).with_suffix(".txt")

    return label_path.as_posix()


# Records details about an invalid image file.
def add_skipped_image_file(
    skipped_image_files: list[dict],
    zip_file_name: str,
    image_name: str,
    reason: str,
):
    skipped_image_files.append(
        {
            "zip_file": zip_file_name,
            "image_file": image_name,
            "reason": reason,
        }
    )


# Checks whether an image file inside a ZIP can be opened properly.
def is_valid_zip_image(
    zip_file: ZipFile,
    zip_file_name: str,
    image_name: str,
    skipped_image_files: list[dict],
) -> bool:
    try:
        image_bytes = zip_file.read(image_name)

        # Verify that the image file is readable.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Open again after verify so the image size can be checked safely.
        with Image.open(BytesIO(image_bytes)) as image:
            image_width, image_height = image.size

        if image_width <= 0 or image_height <= 0:
            add_skipped_image_file(
                skipped_image_files,
                zip_file_name,
                image_name,
                "invalid_image_size",
            )
            return False

        return True

    except UnidentifiedImageError:
        add_skipped_image_file(
            skipped_image_files,
            zip_file_name,
            image_name,
            "unidentified_image_file",
        )
        return False

    except OSError:
        add_skipped_image_file(
            skipped_image_files,
            zip_file_name,
            image_name,
            "corrupted_or_unreadable_image",
        )
        return False

    except Exception as error:
        add_skipped_image_file(
            skipped_image_files,
            zip_file_name,
            image_name,
            f"image_check_error: {error}",
        )
        return False


# Finds all valid image-label pairs inside one ZIP file.
def discover_image_label_pairs(
    zip_file_path: Path,
    skipped_image_files: list[dict],
) -> list[dict]:
    image_label_pairs = []

    with ZipFile(zip_file_path) as zip_file:
        zip_names = set(zip_file.namelist())

        for zip_entry in zip_file.infolist():
            if zip_entry.is_dir():
                continue

            image_path = PurePosixPath(zip_entry.filename)

            if should_skip_zip_entry(image_path):
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            split_and_index = find_split_and_images_index(image_path)

            if split_and_index is None:
                continue

            dataset_split, images_index = split_and_index
            label_name = get_matching_label_name(zip_entry.filename, images_index)

            # Stop if an image does not have its matching label file.
            if label_name not in zip_names:
                raise FileNotFoundError(
                    f"Missing label for image in {zip_file_path.name}: {zip_entry.filename}"
                )

            # Skip the image and its label if the image file is invalid.
            if not is_valid_zip_image(
                zip_file,
                zip_file_path.name,
                zip_entry.filename,
                skipped_image_files,
            ):
                continue

            image_label_pairs.append(
                {
                    "split": dataset_split,
                    "image_name": zip_entry.filename,
                    "label_name": label_name,
                    "image_extension": image_path.suffix.lower(),
                }
            )

    return sorted(
        image_label_pairs,
        key=lambda image_label_pair: (
            image_label_pair["split"],
            image_label_pair["image_name"].lower(),
        ),
    )


# Records details about an invalid annotation row.
def add_skipped_label_row(
    skipped_label_rows: list[dict],
    zip_file_name: str,
    label_name: str,
    line_number: int,
    reason: str,
    line: str,
):
    skipped_label_rows.append(
        {
            "zip_file": zip_file_name,
            "label_file": label_name,
            "line_number": line_number,
            "reason": reason,
            "line": line,
        }
    )


# Converts all valid YOLO annotation rows into one class called fire_smoke.
def convert_label_text_to_single_class(
    label_text: str,
    zip_file_name: str,
    label_name: str,
    skipped_label_rows: list[dict],
) -> str:
    converted_lines = []

    for line_number, line in enumerate(label_text.splitlines(), start=1):
        stripped_line = line.strip()

        if not stripped_line:
            continue

        label_parts = stripped_line.split()

        # A valid YOLO row must have class, x_center, y_center, width, and height.
        if len(label_parts) != 5:
            add_skipped_label_row(
                skipped_label_rows,
                zip_file_name,
                label_name,
                line_number,
                "wrong_column_count",
                line,
            )
            continue

        _, x_center, y_center, width, height = label_parts

        # Convert bounding-box values to numbers.
        try:
            x_center_number = float(x_center)
            y_center_number = float(y_center)
            width_number = float(width)
            height_number = float(height)
        except ValueError:
            add_skipped_label_row(
                skipped_label_rows,
                zip_file_name,
                label_name,
                line_number,
                "non_numeric_value",
                line,
            )
            continue

        # YOLO bounding-box values must be normalized between 0 and 1.
        if not (
            0 <= x_center_number <= 1
            and 0 <= y_center_number <= 1
            and 0 < width_number <= 1
            and 0 < height_number <= 1
        ):
            add_skipped_label_row(
                skipped_label_rows,
                zip_file_name,
                label_name,
                line_number,
                "out_of_range_box",
                line,
            )
            continue

        # Replace the original class ID with 0 for fire_smoke.
        converted_lines.append(f"0 {x_center} {y_center} {width} {height}")

    if not converted_lines:
        return ""

    return "\n".join(converted_lines) + "\n"


# Clears the output folder if it exists, then creates a fresh YOLO folder structure.
def create_clean_dataset_folders(dataset_path: Path):
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    for dataset_split in DATASET_SPLITS:
        (dataset_path / dataset_split / "images").mkdir(parents=True)
        (dataset_path / dataset_split / "labels").mkdir(parents=True)


# Writes the YOLO data.yaml file for the combined dataset.
def write_data_yaml(dataset_path: Path):
    data_yaml_text = "\n".join(
        [
            f"path: {dataset_path.as_posix()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "",
            "nc: 1",
            f"names: ['{FINAL_CLASS_NAME}']",
            "",
        ]
    )

    (dataset_path / "data.yaml").write_text(data_yaml_text, encoding="utf-8")


# Counts files with selected extensions inside a folder.
def count_files_with_extensions(folder_path: Path, allowed_extensions: set[str]) -> int:
    if not folder_path.exists():
        return 0

    return sum(
        1
        for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
    )


# Prints the number of images and labels in the final dataset.
def print_dataset_directory_counts(dataset_path: Path):
    total_images = 0
    total_labels = 0

    print("\nDataset directory counts:")

    for dataset_split in DATASET_SPLITS:
        images_folder_path = dataset_path / dataset_split / "images"
        labels_folder_path = dataset_path / dataset_split / "labels"

        image_count = count_files_with_extensions(images_folder_path, IMAGE_EXTENSIONS)
        label_count = count_files_with_extensions(labels_folder_path, {".txt"})

        total_images += image_count
        total_labels += label_count

        print(f"{dataset_split}/images: {image_count}")
        print(f"{dataset_split}/labels: {label_count}")

    print(f"all images: {total_images}")
    print(f"all labels: {total_labels}")


# Creates the full combined YOLO dataset from the original ZIP files.
def create_combined_dataset():
    zip_file_paths = sorted(ORIGINAL_ZIP_FOLDER_PATH.glob("*.zip"))

    if not zip_file_paths:
        raise FileNotFoundError(f"No zip files found in: {ORIGINAL_ZIP_FOLDER_PATH}")

    print("Finding image-label pairs in original zip files...")

    discovered_pairs_by_zip = {}
    expected_counts = {dataset_split: 0 for dataset_split in DATASET_SPLITS}
    skipped_image_files = []

    # First pass: find all valid image-label pairs in each ZIP file.
    for zip_file_path in zip_file_paths:
        image_label_pairs = discover_image_label_pairs(zip_file_path, skipped_image_files)
        discovered_pairs_by_zip[zip_file_path] = image_label_pairs

        for image_label_pair in image_label_pairs:
            expected_counts[image_label_pair["split"]] += 1

        print(f"- {zip_file_path.name}: {len(image_label_pairs)} valid pairs")

    print("\nCreating clean combined dataset...")

    # Clear and recreate the final dataset folder directly.
    create_clean_dataset_folders(COMBINED_DATASET_PATH)

    written_counts = {dataset_split: 0 for dataset_split in DATASET_SPLITS}
    skipped_label_rows = []

    # Second pass: copy images, convert labels, and write them into the new dataset.
    for dataset_index, zip_file_path in enumerate(zip_file_paths, start=1):
        print(f"Copying dataset {dataset_index:02d}: {zip_file_path.name}")

        with ZipFile(zip_file_path) as zip_file:
            for image_label_pair in discovered_pairs_by_zip[zip_file_path]:
                dataset_split = image_label_pair["split"]
                written_counts[dataset_split] += 1

                # Create a new unique file name to avoid duplicate names.
                file_stem = f"{dataset_split}_{written_counts[dataset_split]:05d}"

                image_output_path = (
                    COMBINED_DATASET_PATH
                    / dataset_split
                    / "images"
                    / f"{file_stem}{image_label_pair['image_extension']}"
                )

                label_output_path = (
                    COMBINED_DATASET_PATH
                    / dataset_split
                    / "labels"
                    / f"{file_stem}.txt"
                )

                # Copy the valid image from the ZIP file into the new dataset folder.
                with zip_file.open(image_label_pair["image_name"]) as source_image_file:
                    with image_output_path.open("wb") as output_image_file:
                        shutil.copyfileobj(source_image_file, output_image_file)

                # Read, clean, and convert the matching YOLO label file.
                label_text = zip_file.read(image_label_pair["label_name"]).decode(
                    "utf-8",
                    errors="replace",
                )

                converted_label_text = convert_label_text_to_single_class(
                    label_text,
                    zip_file_path.name,
                    image_label_pair["label_name"],
                    skipped_label_rows,
                )

                label_output_path.write_text(converted_label_text, encoding="utf-8")

    # Make sure the number of written files matches what was discovered.
    if written_counts != expected_counts:
        raise RuntimeError(
            f"Written counts do not match expected counts: {written_counts} != {expected_counts}"
        )

    # Write data.yaml in the final dataset folder.
    write_data_yaml(COMBINED_DATASET_PATH)

    print("\nDone. Combined dataset was created successfully.")
    print(f"train: {written_counts['train']}")
    print(f"val:   {written_counts['val']}")
    print(f"test:  {written_counts['test']}")
    print(f"skipped invalid image files: {len(skipped_image_files)}")
    print(f"skipped invalid annotation rows: {len(skipped_label_rows)}")
    print(f"yaml:  {COMBINED_DATASET_PATH / 'data.yaml'}")

    # Print the final image and label counts.
    print_dataset_directory_counts(COMBINED_DATASET_PATH)


# Runs the dataset creation process when the file is executed directly.
if __name__ == "__main__":
    create_combined_dataset()