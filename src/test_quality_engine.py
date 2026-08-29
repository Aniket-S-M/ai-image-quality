from pathlib import Path

from inference import predict_image
from quality_engine import assess_image
from PIL import Image


IMAGE_DIR = Path(
    "data/generated/dataset_expanded/test"
)

TEST_IMAGES = [
    IMAGE_DIR / "000000001584_blur_mild.jpg",
    IMAGE_DIR / "000000001584_blur_moderate.jpg",
    IMAGE_DIR / "000000001584_blur_severe.jpg",
]


print("=" * 70)
print("QUALITY ENGINE SEVERITY TEST")
print("=" * 70)

for image_path in TEST_IMAGES:

    image = Image.open(image_path)

    prediction = predict_image(image)

    assessment = assess_image(
        image_path=image_path,
        prediction=prediction["prediction"],
        confidence=prediction["confidence"],
    )

    print()
    print("-" * 70)

    print(
        f"Image       : {image_path.name}"
    )

    print(
        f"Expected    : "
        f"{image_path.stem.split('_')[-1]}"
    )

    print(
        f"CNN issue   : "
        f"{prediction['prediction']}"
    )

    print(
        f"CNN conf.   : "
        f"{prediction['confidence']:.4f}"
    )

    print(
        f"Score       : "
        f"{assessment['quality_score']}"
    )

    print(
        f"Label       : "
        f"{assessment['quality_label']}"
    )

    print(
        f"Issues      : "
        f"{assessment['issues']}"
    )


print()
print("=" * 70)
print("SEVERITY TEST COMPLETE")
print("=" * 70)