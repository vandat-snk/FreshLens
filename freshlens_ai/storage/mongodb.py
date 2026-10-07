"""MongoDB connection helpers for the active FreshLens pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load repository-local environment variables when available.
load_dotenv(PROJECT_ROOT / ".env")


def _mongo_uri() -> str:
    uri = os.getenv("MONGODB_URI", "").strip()

    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not configured. "
            "Set it in .env or the process environment."
        )

    return uri


def _mongo_db_name() -> str:
    name = os.getenv("MONGODB_DB", "freshlens").strip()

    if not name:
        raise RuntimeError("MONGODB_DB cannot be empty.")

    return name


def get_client(
    uri: str | None = None,
    *,
    timeout_ms: int = 5000,
) -> MongoClient:
    """Create a MongoDB client without performing writes."""

    return MongoClient(
        uri or _mongo_uri(),
        serverSelectionTimeoutMS=timeout_ms,
    )


def get_db(
    uri: str | None = None,
    db_name: str | None = None,
) -> Database:
    """Return the configured FreshLens MongoDB database."""

    return get_client(uri)[
        db_name or _mongo_db_name()
    ]


def test_connection(
    uri: str | None = None,
) -> bool:
    """Ping MongoDB and report a configuration-safe status."""

    client = None

    try:
        client = get_client(uri)
        client.admin.command("ping")

        print("[OK] MongoDB connection successful.")
        return True

    except Exception as exc:
        print(
            "[ERROR] MongoDB connection failed: "
            f"{type(exc).__name__}. "
            "Check .env, Atlas network access and credentials."
        )

        return False

    finally:
        if client is not None:
            client.close()