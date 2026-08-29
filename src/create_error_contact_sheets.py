from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

ERROR_CSV = (
    ROOT
    / "data"
    / "evaluation"
    / "misclassified.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation"
    / "contact_sheets"
)


# ==================================================
# SETTINGS
# ==================================================

THUMBNAIL_WIDTH = 220
THUMBNAIL_HEIGHT = 170

COLUMNS = 4

PADDING = 20

TEXT_HEIGHT = 55


# ==================================================
# FIND IMAGE
# ==================================================

def get_image_path(path_string):

    path = Path(path_string)

    if path.exists():
        return path

    return None


# ==================================================
# CREATE CONTACT SHEET
# ==================================================

def create_contact_sheet(group, actual, predicted):

    images = []

    for _, row in group.iterrows():

        image_path = get_image_path(
            row["image_path"]
        )

        if image_path is None:
            continue

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (
                    THUMBNAIL_WIDTH,
                    THUMBNAIL_HEIGHT
                )
            )

            images.append(
                (
                    image,
                    row
                )
            )

        except Exception as e:

            print(
                f"Could not open {image_path}: {e}"
            )

    if not images:
        return

    rows = (
        len(images) + COLUMNS - 1
    ) // COLUMNS

    sheet_width = (
        COLUMNS * THUMBNAIL_WIDTH
        + (COLUMNS + 1) * PADDING
    )

    sheet_height = (
        rows * (
            THUMBNAIL_HEIGHT
            + TEXT_HEIGHT
            + PADDING
        )
        + PADDING
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    # ----------------------------------------------
    # Title
    # ----------------------------------------------

    title = (
        f"Actual: {actual}  →  "
        f"Predicted: {predicted}"
    )

    draw.text(
        (
            PADDING,
            5
        ),
        title,
        fill="black"
    )

    # ----------------------------------------------
    # Place images
    # ----------------------------------------------

    title_offset = 35

    for index, (image, row) in enumerate(images):

        column = index % COLUMNS

        row_number = index // COLUMNS

        x = (
            PADDING
            + column * (
                THUMBNAIL_WIDTH
                + PADDING
            )
        )

        y = (
            title_offset
            + PADDING
            + row_number * (
                THUMBNAIL_HEIGHT
                + TEXT_HEIGHT
                + PADDING
            )
        )

        # Center thumbnail
        image_x = (
            x
            + (
                THUMBNAIL_WIDTH
                - image.width
            ) // 2
        )

        image_y = (
            y
            + (
                THUMBNAIL_HEIGHT
                - image.height
            ) // 2
        )

        sheet.paste(
            image,
            (
                image_x,
                image_y
            )
        )

        # ------------------------------------------
        # Image information
        # ------------------------------------------

        confidence = row[
            "prediction_confidence"
        ]

        text = (
            f"Confidence: "
            f"{confidence:.2f}\n"
            f"Severity: "
            f"{row['severity']}"
        )

        draw.multiline_text(
            (
                x,
                y + THUMBNAIL_HEIGHT + 5
            ),
            text,
            fill="black"
        )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    filename = (
        f"{actual}_TO_{predicted}.png"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    sheet.save(
        output_path
    )

    print(
        f"Created: {output_path}"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "==================================="
    )

    print(
        "ERROR CONTACT SHEET GENERATION"
    )

    print(
        "==================================="
    )

    if not ERROR_CSV.exists():

        raise FileNotFoundError(
            f"Error CSV not found:\n"
            f"{ERROR_CSV}"
        )

    df = pd.read_csv(
        ERROR_CSV
    )

    print(
        f"Misclassified records: {len(df)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Group errors by actual → predicted
    # ----------------------------------------------

    groups = df.groupby(
        [
            "issue",
            "predicted_issue"
        ]
    )

    for (actual, predicted), group in groups:

        print(
            f"\nProcessing:"
            f" {actual} → {predicted}"
        )

        create_contact_sheet(
            group,
            actual,
            predicted
        )

    print(
        "\n==================================="
    )

    print(
        "CONTACT SHEETS COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()