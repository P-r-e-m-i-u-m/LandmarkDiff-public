"""Tests for validate_landmarks() and LandmarkValidationResult."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.validation import (
    MEDIAPIPE_LANDMARK_COUNT,
    LandmarkValidationResult,
    validate_landmarks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_json(tmp_path, landmarks, confidence=None, filename="face.json"):
    data = {"landmarks": landmarks}
    if confidence is not None:
        data["confidence"] = confidence
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_csv(tmp_path, landmarks, header=True, filename="face.csv"):
    lines = []
    if header:
        lines.append("x,y,z")
    for lm in landmarks:
        lines.append(",".join(str(v) for v in lm))
    p = tmp_path / filename
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _valid_landmarks(n=MEDIAPIPE_LANDMARK_COUNT):
    """Generate n valid 3D normalized landmarks."""
    return [[0.5, 0.5, 0.0]] * n


# ---------------------------------------------------------------------------
# File existence checks
# ---------------------------------------------------------------------------

def test_file_not_found(tmp_path):
    result = validate_landmarks(tmp_path / "missing.json")
    assert not result.valid
    assert any("not found" in e.lower() for e in result.errors)


def test_unsupported_format(tmp_path):
    p = tmp_path / "face.txt"
    p.write_text("hello")
    result = validate_landmarks(p)
    assert not result.valid
    assert any("unsupported" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def test_valid_json_mediapipe(tmp_path):
    p = _make_json(tmp_path, _valid_landmarks())
    result = validate_landmarks(p)
    assert result.valid
    assert result.landmark_count == MEDIAPIPE_LANDMARK_COUNT
    assert result.dimensions == 3


def test_valid_json_bare_list(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(_valid_landmarks()), encoding="utf-8")
    result = validate_landmarks(p)
    assert result.valid


def test_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    result = validate_landmarks(p)
    assert not result.valid
    assert any("json" in e.lower() for e in result.errors)


def test_wrong_landmark_count_json(tmp_path):
    p = _make_json(tmp_path, _valid_landmarks(10))
    result = validate_landmarks(p)
    assert not result.valid
    assert any("expected" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def test_valid_csv_with_header(tmp_path):
    p = _make_csv(tmp_path, _valid_landmarks(), header=True)
    result = validate_landmarks(p)
    assert result.valid
    assert result.landmark_count == MEDIAPIPE_LANDMARK_COUNT


def test_valid_csv_without_header(tmp_path):
    p = _make_csv(tmp_path, _valid_landmarks(), header=False)
    result = validate_landmarks(p)
    assert result.valid


def test_invalid_csv(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("x,y,z\nnot,a,number\n", encoding="utf-8")
    result = validate_landmarks(p)
    assert not result.valid


# ---------------------------------------------------------------------------
# NaN / Inf checks
# ---------------------------------------------------------------------------

def test_nan_values(tmp_path):
    lms = _valid_landmarks()
    lms[0] = [float("nan"), 0.5, 0.0]
    p = _make_json(tmp_path, lms)
    result = validate_landmarks(p)
    assert not result.valid
    assert any("nan" in e.lower() for e in result.errors)


def test_inf_values(tmp_path):
    lms = _valid_landmarks()
    lms[0] = [float("inf"), 0.5, 0.0]
    p = _make_json(tmp_path, lms)
    result = validate_landmarks(p)
    assert not result.valid
    assert any("inf" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Out of bounds
# ---------------------------------------------------------------------------

def test_out_of_bounds_warning(tmp_path):
    lms = _valid_landmarks()
    lms[0] = [1.5, 0.5, 0.0]  # x > 1.0
    p = _make_json(tmp_path, lms)
    result = validate_landmarks(p)
    assert result.valid  # warning only, not error
    assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

def test_low_confidence_warning(tmp_path):
    lms = _valid_landmarks()
    confidence = [0.9] * MEDIAPIPE_LANDMARK_COUNT
    confidence[0] = 0.1  # below threshold
    p = _make_json(tmp_path, lms, confidence=confidence)
    result = validate_landmarks(p, min_confidence=0.5)
    assert result.valid
    assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# LandmarkValidationResult str
# ---------------------------------------------------------------------------

def test_str_valid():
    r = LandmarkValidationResult(valid=True, landmark_count=478, dimensions=3)
    s = str(r)
    assert "VALID" in s
    assert "478" in s


def test_str_invalid():
    r = LandmarkValidationResult(valid=False, errors=["File not found"])
    s = str(r)
    assert "INVALID" in s
    assert "ERROR" in s