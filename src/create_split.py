from pathlib import Path
import csv
import random


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "metadata" / "clean_images.csv"
SPLIT_DIR = ROOT / "data" / "splits"

SEED = 42


def main():

    # -----------------------------------------
    # Check input
    # -----------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {INPUT_FILE}"
        )

    # -----------------------------------------
    # Read only valid images
    # -----------------------------------------

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        images = [
            row for row in reader
            if row["valid"].lower() == "true"
        ]

    print(f"Valid source images: {len(images)}")

    if len(images) < 10:
        raise RuntimeError(
            "Not enough images to create a split."
        )

    # -----------------------------------------
    # Reproducible shuffle
    # -----------------------------------------

    rng = random.Random(SEED)
    rng.shuffle(images)

    # -----------------------------------------
    # 70 / 15 / 15 split
    # -----------------------------------------

    total = len(images)

    train_count = int(total * 0.70)
    val_count = int(total * 0.15)

    train = images[:train_count]

    val = images[
        train_count:
        train_count + val_count
    ]

    test = images[
        train_count + val_count:
    ]

    # -----------------------------------------
    # Create split directory
    # -----------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # Save splits
    # -----------------------------------------

    for split_name, split_data in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:

        output_file = (
            SPLIT_DIR /
            f"{split_name}.csv"
        )

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=images[0].keys()
            )

            writer.writeheader()
            writer.writerows(split_data)

        print(
            f"{split_name:5s}: "
            f"{len(split_data)} images"
        )

    # -----------------------------------------
    # Verify no overlap
    # -----------------------------------------

    train_ids = {
        row["source_image_id"]
        for row in train
    }

    val_ids = {
        row["source_image_id"]
        for row in val
    }

    test_ids = {
        row["source_image_id"]
        for row in test
    }

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nNo source-image overlap detected.")
    print("Source-level split created successfully.")


if __name__ == "__main__":
    main()