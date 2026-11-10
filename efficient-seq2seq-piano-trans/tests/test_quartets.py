"""Quartets import and explicit handling of inconsistent reference rhythms."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from a2s.import_data import quartets, saxophone
from a2s.kern import validate
from a2s.tokenizer import KernFieldTokenizer


def test_quartets_preserves_pairs_partitions_and_saxophone_defaults(tmp_path):
    for directory in ("wav", "krn_clean", "a2s_partitions/quartets"):
        (tmp_path / directory).mkdir(parents=True)
    for split in ("train", "val", "test"):
        sf.write(tmp_path / "wav" / f"{split}.wav", np.zeros(1600), 16000)
        (tmp_path / "krn_clean" / f"{split}.krn").write_text(
            "**kern\t**kern\t**kern\t**kern\n4c\t4e\t4g\t4cc\n*-\t*-\t*-\t*-\n")
        pair = f"{split}.wav\t{split}.krn\n"
        for directory in ("a2s_partitions", "a2s_partitions/quartets"):
            (tmp_path / directory / f"{split}.txt").write_text(pair)
    imported = quartets(tmp_path, tmp_path / "quartet-output")
    rows = [json.loads(line) for line in imported.read_text().splitlines()]
    assert [row["split"] for row in rows] == ["train", "validation", "test"]
    assert {row["category"] for row in rows} == {"fixed"}
    assert len({row["id"] for row in rows}) == 3
    for row in rows:
        assert Path(row["audio"]).is_file() and Path(row["score"]).is_file()
        assert row["start"] == 0 and row["end"] == 0.1
    previous = saxophone(tmp_path, tmp_path / "sax-output")
    sax_rows = [json.loads(line) for line in previous.read_text().splitlines()]
    assert {row["category"] for row in sax_rows} == {"mono"}
    assert [{k: v for k, v in row.items() if k != "category"} for row in rows] == [
        {k: v for k, v in row.items() if k != "category"} for row in sax_rows]


def test_inconsistent_reference_is_reported_without_altering_tokens():
    text = "**kern\t**kern\n4r\t8c\n2r\t8d\n*-\t*-\n"
    with pytest.raises(ValueError, match="overlaps"):
        validate(text)
    report = validate(text, strict_rhythm=False)
    assert not report["structural_rhythm_valid"]
    assert report["quarter_length"] is None
    assert any("overlaps" in issue for issue in report["rhythm_warnings"])
    tokenizer = KernFieldTokenizer.fit([text])
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_implicit_reference_duration_is_reported_without_rewriting():
    text = "**kern\n4c\nr\n*-\n"
    with pytest.raises(ValueError, match="Missing explicit reciprocal"):
        validate(text)
    report = validate(text, strict_rhythm=False)
    assert not report["structural_rhythm_valid"]
    assert report["quarter_length"] is None
    assert any("Missing explicit reciprocal duration: r" in issue
               for issue in report["rhythm_warnings"])
    tokenizer = KernFieldTokenizer.fit([text])
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize("text", [
    "**kern\n*^\n4c\n*-\n",             # Wrong field count.
    "**kern\n4c\n",                      # Missing terminator.
    "**kern\nxyz\n*-\n",                # Invalid pitch/duration.
    "**kern\n4c 8e\n*-\n",              # Invalid chord notation.
    "**kern\n.\n*-\n",                  # No explicit events.
    "**kern\t**text\n4c\tword\n*-\t*-\n",  # Unsupported spine type.
])
def test_reference_mode_still_rejects_malformed_structure(text):
    with pytest.raises(ValueError):
        validate(text, strict_rhythm=False)


@pytest.mark.parametrize("name", ["mono", "fixed", "open"])
def test_valid_reference_keeps_strict_report(name):
    root = Path(__file__).resolve().parents[1]
    text = (root / "examples/kern" / f"{name}.krn").read_text()
    assert validate(text, strict_rhythm=False) == validate(text)
