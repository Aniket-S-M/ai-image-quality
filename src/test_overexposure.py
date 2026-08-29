from pathlib import Path

import cv2

from degradation import apply_overexposure


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "clean"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "generated"
    / "overexposure_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_brightness(image):
    """
    Mean grayscale intensity.
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale.mean()


def calculate_highlight_clipping(image):
    """
    Percentage of pixels at or near maximum intensity.

    A high value indicates loss of highlight detail.
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clipped_pixels = (
        grayscale >= 250
    ).sum()

    total_pixels = grayscale.size

    percentage = (
        clipped_pixels /
        total_pixels
    ) * 100

    return percentage


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
# Original measurements
# -----------------------------------------

original_brightness = calculate_brightness(
    image
)

original_clipping = calculate_highlight_clipping(
    image
)

print(
    f"\nOriginal brightness: "
    f"{original_brightness:.2f}"
)

print(
    f"Original highlight clipping: "
    f"{original_clipping:.2f}%"
)


# -----------------------------------------
# Generate overexposed versions
# -----------------------------------------

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    brightened = apply_overexposure(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_overexposure_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        brightened
    )

    brightness = calculate_brightness(
        brightened
    )

    clipping = calculate_highlight_clipping(
        brightened
    )

    print(
        f"{severity.capitalize():10s} "
        f"brightness: {brightness:.2f} | "
        f"clipping: {clipping:.2f}%"
    )


print(
    "\nOverexposure test complete."
)