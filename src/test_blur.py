from pathlib import Path

import cv2

from degradation import apply_blur


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "clean"
OUTPUT_DIR = ROOT / "data" / "generated" / "blur_test"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_sharpness(image):
    """
    Calculate image sharpness using
    variance of the Laplacian.

    Higher value  -> sharper image
    Lower value   -> blurrier image
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_64F
    )

    variance = laplacian.var()

    return variance


# -----------------------------------------
# Load original image
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
# Measure original sharpness
# -----------------------------------------

original_sharpness = calculate_sharpness(
    image
)

print(
    f"\nOriginal sharpness: "
    f"{original_sharpness:.2f}"
)


# -----------------------------------------
# Generate and measure blur
# -----------------------------------------

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    blurred = apply_blur(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_blur_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        blurred
    )

    sharpness = calculate_sharpness(
        blurred
    )

    print(
        f"{severity.capitalize():10s} "
        f"sharpness: {sharpness:.2f}"
    )


print("\nBlur + sharpness test complete.")