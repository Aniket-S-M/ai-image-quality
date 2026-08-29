from pathlib import Path

import cv2

from degradation import apply_underexposure


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "clean"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "generated"
    / "underexposure_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_brightness(image):
    """
    Calculate mean grayscale intensity.

    Higher value -> brighter image
    Lower value  -> darker image
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale.mean()


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
# Original brightness
# -----------------------------------------

original_brightness = calculate_brightness(
    image
)

print(
    f"\nOriginal brightness: "
    f"{original_brightness:.2f}"
)


# -----------------------------------------
# Generate underexposed versions
# -----------------------------------------

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    darkened = apply_underexposure(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_underexposure_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        darkened
    )

    brightness = calculate_brightness(
        darkened
    )

    print(
        f"{severity.capitalize():10s} "
        f"brightness: {brightness:.2f}"
    )


print(
    "\nUnderexposure test complete."
)