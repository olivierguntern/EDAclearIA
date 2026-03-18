"""Tests unitaires pour le module analyzer."""

import io
from pathlib import Path

import pandas as pd
import pytest

from src.analyzer import analyze


def test_analyze_runs(tmp_path: Path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b,c\n1,2,3\n4,5,6\n")
    analyze(csv)  # ne doit pas lever d'exception


def test_analyze_missing_values(tmp_path: Path, capsys):
    csv = tmp_path / "missing.csv"
    csv.write_text("a,b\n1,\n2,3\n")
    analyze(csv)
    output = capsys.readouterr().out
    assert "b" in output  # colonne avec valeur manquante signalée
