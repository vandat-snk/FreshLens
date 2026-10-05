"""Small testable helpers for binding Streamlit results to the current image."""

from __future__ import annotations

from freshlens_ai.utils.file_io import sha256_bytes


RESULT_KEY = "last_result"
IMAGE_HASH_KEY = "last_result_image_sha256"


def image_identity(image_bytes: bytes) -> str:
    """Return a stable identity for the exact uploaded/captured image bytes."""
    return sha256_bytes(image_bytes)


def store_result(session_state, image_bytes: bytes, result) -> None:
    """Store a result together with the image it belongs to."""
    session_state[RESULT_KEY] = result
    session_state[IMAGE_HASH_KEY] = image_identity(image_bytes)


def clear_result(session_state) -> None:
    """Remove any previous analysis result."""
    session_state.pop(RESULT_KEY, None)
    session_state.pop(IMAGE_HASH_KEY, None)


def result_for_image(session_state, image_bytes: bytes):
    """Return the cached result only when it belongs to the current image."""
    if (
        session_state.get(IMAGE_HASH_KEY)
        != image_identity(image_bytes)
    ):
        return None

    return session_state.get(RESULT_KEY)
