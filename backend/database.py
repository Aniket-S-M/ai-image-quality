import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# DATABASE PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = ROOT / "data"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_FILE = DATABASE_DIR / "quality_analysis.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_database():

    connection = get_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                quality_score INTEGER NOT NULL,
                quality_label TEXT NOT NULL,
                issues TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# CREATE ANALYSIS
# ============================================================

def save_analysis(
    filename,
    quality_score,
    quality_label,
    issues,
):

    connection = get_connection()

    try:

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        cursor = connection.execute(
            """
            INSERT INTO analyses (
                filename,
                quality_score,
                quality_label,
                issues,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                quality_score,
                quality_label,
                json.dumps(issues),
                created_at,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# ============================================================
# GET ALL ANALYSES
# ============================================================

def get_all_analyses():

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                quality_score,
                quality_label,
                issues,
                created_at
            FROM analyses
            ORDER BY id DESC
            """
        ).fetchall()

        results = []

        for row in rows:

            results.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "quality_score": row[
                        "quality_score"
                    ],
                    "quality_label": row[
                        "quality_label"
                    ],
                    "issues": json.loads(
                        row["issues"]
                    ),
                    "created_at": row[
                        "created_at"
                    ],
                }
            )

        return results

    finally:

        connection.close()


# ============================================================
# GET ONE ANALYSIS
# ============================================================

def get_analysis(
    analysis_id
):

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT
                id,
                filename,
                quality_score,
                quality_label,
                issues,
                created_at
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()

        if row is None:

            return None

        return {
            "id": row["id"],
            "filename": row["filename"],
            "quality_score": row[
                "quality_score"
            ],
            "quality_label": row[
                "quality_label"
            ],
            "issues": json.loads(
                row["issues"]
            ),
            "created_at": row[
                "created_at"
            ],
        }

    finally:

        connection.close()


# ============================================================
# INITIALIZE ON IMPORT
# ============================================================

init_database()