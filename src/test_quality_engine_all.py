from pathlib import Path

from quality_engine import assess_image


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset_expanded"
    / "test"
)


# ============================================================
# FIND TEST IMAGE
# ============================================================

def find_image(issue, severity=None):

    # --------------------------------------------------------
    # CLEAN IMAGE
    # --------------------------------------------------------

    if issue == "none":

        matches = list(
            TEST_DIR.glob("*_clean.jpg")
        )

    # --------------------------------------------------------
    # NOISE
    #
    # Ignore salt-and-pepper noise variants.
    # --------------------------------------------------------

    elif issue == "noise":

        matches = list(
            TEST_DIR.glob("*_noise_*.jpg")
        )

        matches = [
            p
            for p in matches
            if "_noise_sp_" not in p.name
        ]

        if severity is not None:

            severity_matches = [
                p
                for p in matches
                if p.name.endswith(
                    f"_noise_{severity}.jpg"
                )
            ]

            if severity_matches:

                matches = severity_matches

    # --------------------------------------------------------
    # OTHER ISSUES
    # --------------------------------------------------------

    else:

        matches = list(
            TEST_DIR.glob(
                f"*_{issue}_{severity}.jpg"
            )
        )

    if not matches:

        raise FileNotFoundError(
            f"No image found for "
            f"{issue} / {severity}"
        )

    return sorted(matches)[0]


# ============================================================
# TEST ONE IMAGE
# ============================================================

def test_image(
    image_path,
    expected_issue,
    expected_severity,
):

    # --------------------------------------------------------
    # RUN FINAL QUALITY ENGINE
    #
    # Internally:
    #
    # Image
    #   ↓
    # CV features
    #   ↓
    # Random Forest V3
    #   ↓
    # Severity
    #   ↓
    # Isolation Forest
    #   ↓
    # Quality assessment
    # --------------------------------------------------------

    assessment = assess_image(
        image_path
    )

    # --------------------------------------------------------
    # Extract detected result
    # --------------------------------------------------------

    if assessment["issues"]:

        detected_issue = (
            assessment["issues"][0]["type"]
        )

        detected_severity = (
            assessment["issues"][0]["severity"]
        )

        confidence = (
            assessment["issues"][0]["confidence"]
        )

    else:

        detected_issue = "none"
        detected_severity = "none"
        confidence = 0.0

    # --------------------------------------------------------
    # Check expected result
    # --------------------------------------------------------

    issue_ok = (
        detected_issue
        == expected_issue
    )

    severity_ok = (
        detected_severity
        == expected_severity
        if expected_issue != "none"
        else detected_severity == "none"
    )

    status = (
        "PASS"
        if issue_ok and severity_ok
        else "CHECK"
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("-" * 72)

    print(
        f"Image          : "
        f"{image_path.name}"
    )

    print(
        f"Expected issue : "
        f"{expected_issue}"
    )

    print(
        f"Expected level : "
        f"{expected_severity}"
    )

    print(
        f"Detected issue : "
        f"{detected_issue}"
    )

    print(
        f"Confidence     : "
        f"{confidence:.4f}"
    )

    print(
        f"Severity       : "
        f"{detected_severity}"
    )

    print(
        f"Quality score  : "
        f"{assessment['quality_score']}"
    )

    print(
        f"Quality label  : "
        f"{assessment['quality_label']}"
    )

    print(
        f"Result         : "
        f"{status}"
    )

    return status == "PASS"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FULL QUALITY ENGINE TEST")
    print("=" * 72)

    tests = [

        # ----------------------------------------------------
        # BLUR
        # ----------------------------------------------------

        ("blur", "mild"),
        ("blur", "moderate"),
        ("blur", "severe"),

        # ----------------------------------------------------
        # NOISE
        # ----------------------------------------------------

        ("noise", "mild"),
        ("noise", "moderate"),
        ("noise", "severe"),

        # ----------------------------------------------------
        # CORRUPTION
        # ----------------------------------------------------

        ("corruption", "mild"),
        ("corruption", "moderate"),
        ("corruption", "severe"),

        # ----------------------------------------------------
        # OVEREXPOSURE
        # ----------------------------------------------------

        ("overexposure", "mild"),
        ("overexposure", "moderate"),
        ("overexposure", "severe"),

        # ----------------------------------------------------
        # UNDEREXPOSURE
        # ----------------------------------------------------

        ("underexposure", "mild"),
        ("underexposure", "moderate"),
        ("underexposure", "severe"),
    ]

    passed = 0
    total = 0
    errors = 0

    # ========================================================
    # DEGRADATION TESTS
    # ========================================================

    for issue, severity in tests:

        try:

            image_path = find_image(
                issue,
                severity
            )

            result = test_image(
                image_path,
                issue,
                severity
            )

            if result:

                passed += 1

            total += 1

        except Exception as e:

            print()
            print("-" * 72)

            print(
                f"[ERROR] "
                f"{issue} / {severity}"
            )

            print(
                f"Reason: {e}"
            )

            total += 1
            errors += 1

    # ========================================================
    # CLEAN IMAGE
    # ========================================================

    print()
    print("-" * 72)
    print("CLEAN IMAGE TEST")
    print("-" * 72)

    try:

        clean_image = find_image(
            "none"
        )

        assessment = assess_image(
            clean_image
        )

        if assessment["issues"]:

            detected_issue = (
                assessment["issues"][0]["type"]
            )

            detected_severity = (
                assessment["issues"][0]["severity"]
            )

        else:

            detected_issue = "none"
            detected_severity = "none"

        clean_pass = (
            detected_issue == "none"
            and detected_severity == "none"
            and assessment["quality_label"]
            == "ACCEPTABLE"
        )

        if clean_pass:

            passed += 1

        else:

            errors += 1

        total += 1

        print(
            f"Image          : "
            f"{clean_image.name}"
        )

        print(
            f"Expected issue : none"
        )

        print(
            f"Detected issue : "
            f"{detected_issue}"
        )

        print(
            f"Severity       : "
            f"{detected_severity}"
        )

        print(
            f"Quality score  : "
            f"{assessment['quality_score']}"
        )

        print(
            f"Quality label  : "
            f"{assessment['quality_label']}"
        )

        print(
            f"Issues         : "
            f"{assessment['issues']}"
        )

        print(
            f"Result         : "
            f"{'PASS' if clean_pass else 'CHECK'}"
        )

    except Exception as e:

        print()
        print(
            f"[ERROR] Clean image test: {e}"
        )

        total += 1
        errors += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("FULL QUALITY ENGINE TEST COMPLETE")
    print("=" * 72)

    print(
        f"Tests passed : "
        f"{passed}/{total}"
    )

    print(
        f"Tests to check : "
        f"{total - passed}/{total}"
    )

    print(
        f"Errors       : "
        f"{errors}"
    )

    print()

    if passed == total:

        print(
            "ALL TESTS PASSED."
        )

        print(
            "ML/CV QUALITY ENGINE READY TO FREEZE."
        )

    else:

        print(
            "Some cases need investigation."
        )

        print(
            "Review the CHECK cases above."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()