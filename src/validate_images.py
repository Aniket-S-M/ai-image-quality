from pathlib import Path
import csv

from PIL import Image, ImageStat


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "clean"
METADATA_DIR = ROOT / "data" / "metadata"

OUTPUT_FILE = METADATA_DIR / "clean_images.csv"


# --------------------------------------------------
# Basic validation rules
# --------------------------------------------------

MIN_WIDTH = 256
MIN_HEIGHT = 192

# Extremely low standard deviation usually means
# the image is almost completely blank.
BLANK_STD_THRESHOLD = 3.0


# --------------------------------------------------
# Validate one image
# --------------------------------------------------

def validate_image(image_path):

    result = {
        "filename": image_path.name,
        "source_image_id": image_path.stem,
        "valid": False,
        "format": "",
        "width": "",
        "height": "",
        "mode": "",
        "mean_intensity": "",
        "intensity_std": "",
        "reason": ""
    }

    try:

        # ------------------------------------------
        # First pass: verify image integrity
        # ------------------------------------------

        with Image.open(image_path) as image:
            image.verify()

        # ------------------------------------------
        # Second pass: actually read the image
        # ------------------------------------------

        with Image.open(image_path) as image:

            image_format = image.format

            rgb_image = image.convert("RGB")

            width, height = rgb_image.size

            # Convert to grayscale for simple statistics
            grayscale = rgb_image.convert("L")

            statistics = ImageStat.Stat(grayscale)

            mean_intensity = statistics.mean[0]
            intensity_std = statistics.stddev[0]

        # ------------------------------------------
        # Store extracted information
        # ------------------------------------------

        result["format"] = image_format
        result["width"] = width
        result["height"] = height
        result["mode"] = "RGB"
        result["mean_intensity"] = round(mean_intensity, 3)
        result["intensity_std"] = round(intensity_std, 3)

        # ------------------------------------------
        # Validation checks
        # ------------------------------------------

        if width < MIN_WIDTH or height < MIN_HEIGHT:

            result["reason"] = "too_small"

        elif intensity_std < BLANK_STD_THRESHOLD:

            result["reason"] = "near_blank"

        else:

            result["valid"] = True
            result["reason"] = "ok"

    except Exception as error:

        result["reason"] = (
            f"unreadable: {type(error).__name__}"
        )

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if not images:

        raise SystemExit(
            f"No JPG images found in {IMAGE_DIR}"
        )

    print(f"Images found: {len(images)}")
    print("Validating...\n")

    results = []

    for image_path in images:

        result = validate_image(image_path)

        results.append(result)

        status = "VALID" if result["valid"] else "INVALID"

        print(
            f"{image_path.name} -> "
            f"{status} ({result['reason']})"
        )

    # ------------------------------------------
    # Save metadata
    # ------------------------------------------

    fieldnames = list(results[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    valid_count = sum(
        result["valid"]
        for result in results
    )

    invalid_count = len(results) - valid_count

    print("\n-----------------------------")
    print(f"Total images : {len(results)}")
    print(f"Valid        : {valid_count}")
    print(f"Invalid      : {invalid_count}")
    print(f"Metadata     : {OUTPUT_FILE}")
    print("-----------------------------")


if __name__ == "__main__":
    main()