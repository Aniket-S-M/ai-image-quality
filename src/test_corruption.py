from pathlib import Path

import cv2

from degradation import apply_jpeg_corruption


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "clean"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "generated"
    / "corruption_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_mean_difference(original, corrupted):
    """
    Calculate the average absolute pixel difference
    between the original and corrupted image.
    """

    difference = cv2.absdiff(
        original,
        corrupted
    )

    return difference.mean()


# -----------------------------------------
# Load one source image
# -----------------------------------------

image_path = sorted(
    INPUT_DIR.glob("*.jpg")
)[0]

print("Input image:")
print(image_path)

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )


# -----------------------------------------
# Generate corrupted versions
# -----------------------------------------

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    corrupted = apply_jpeg_corruption(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_corruption_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        corrupted
    )

    difference = calculate_mean_difference(
        image,
        corrupted
    )

    print(
        f"{severity.capitalize():10s} "
        f"mean pixel difference: "
        f"{difference:.2f}"
    )


print("\nJPEG corruption test complete.")