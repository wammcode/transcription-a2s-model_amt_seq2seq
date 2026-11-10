"""Regression checks for partition integrity and reference-free MIDI generation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("a2m_pipeline", Path(__file__).with_name("pipeline.py"))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def corpus(tmp_path, prefixed):
    for directory in ("wav", "midi", "partitions"):
        (tmp_path / directory).mkdir()
    for split in ("train", "val", "test"):
        (tmp_path / "wav" / f"{split}.wav").write_bytes(b"wav")
        (tmp_path / "midi" / f"{split}.mid").write_bytes(b"midi")
        pair = f"{split}.wav\t{split}.mid"
        if prefixed:
            pair = f"wav/{split}.wav\tmidi/{split}.mid"
        (tmp_path / "partitions" / f"{split}.txt").write_text(pair + "\n")
    return tmp_path


@pytest.mark.parametrize("prefixed", [False, True])
def test_original_partition_formats_and_overlap(tmp_path, prefixed):
    root = corpus(tmp_path, prefixed)
    rows = pipeline.read_partitions(root, "partitions")
    assert [row["split"] for row in rows] == ["train", "val", "test"]
    assert [row["audio_relative"] for row in rows] == ["wav/train.wav", "wav/val.wav", "wav/test.wav"]
    (root / "partitions/test.txt").write_text((root / "partitions/train.txt").read_text())
    with pytest.raises(ValueError, match="overlap"):
        pipeline.read_partitions(root, "partitions")


def test_missing_pair_is_an_error(tmp_path):
    root = corpus(tmp_path, False)
    (root / "midi/val.mid").unlink()
    with pytest.raises(ValueError, match="Missing"):
        pipeline.read_partitions(root, "partitions")


def test_short_audio_and_bounded_cache(tmp_path):
    """Exercise the real legacy dataset with a clip shorter than n_frames / 4."""
    import h5py
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/wav>short.mid").write_bytes(b"placeholder")
    (tmp_path / "cache/wav>short.midi-notes.tsv").write_text(
        "type\ttype_id\tonset_sec\tduration_sec\toffset_sec\tpitch\tvelocity\n"
        "note\t1\t0.01\t0.05\t0.06\t60\t80\n")
    with h5py.File(tmp_path / "audio.h5", "w") as h5:
        group = h5.create_group("0")
        group["audio"] = np.zeros(1600, dtype=np.float32)
        group["duration"] = 0.1
        group["path"] = "short.wav"
    with initialize_config_dir(config_dir=str(Path.cwd() / "config"), version_base=None):
        config = compose(config_name="main_config")
    config.data.dataset_dirs = [str(tmp_path)]
    dataset = pipeline.make_dataset(config, [{"split": "train", "id": 0,
        "samples": 1600, "audio_relative": "wav/short.wav"}], "train")
    sample = dataset[0]
    assert sample["inputs"].numel() == config.data.n_frames * config.data.hop_length
    assert sample["decoder_targets_len"].item() > 0
    assert len(dataset.cache) == 1
    for source in dataset.cache.values():
        source.audio_h5.close()


def test_audio_only_generation_preserves_truncated_and_empty_outputs():
    import torch
    from omegaconf import OmegaConf
    from data.constants import sm_tokenizer
    from data.symbolic_music_tokenizer import TokenOnset, TokenPitch, TokenNoteOff, TokenVel

    config = OmegaConf.create({"data": {"n_frames": 4, "hop_length": 2,
        "sample_rate": 8, "max_token_length": 8}, "training": {"batch_inference": 1}})
    class Frontend:
        def __call__(self, wave):
            return wave[:, None, :]

    class Model:
        def __init__(self):
            self.calls = 0

        def generate(self, features, target_seq_length, berak_on_eos):
            assert tuple(features.shape) == (1, 7, 1)
            self.calls += 1
            if self.calls == 1:
                return torch.tensor([[TokenOnset(0).get_token_val(), TokenPitch(60).get_token_val(),
                                      TokenVel(40).get_token_val(), sm_tokenizer.EOS]])
            # No EOS: retain the sequence and link its NoteOff to the earlier NoteOn.
            return torch.tensor([[TokenOnset(0).get_token_val(), TokenNoteOff(60).get_token_val(),
                                  sm_tokenizer.BLANK]])

    notes, truncated = pipeline.predict_audio(Model(), Frontend(), np.zeros(16, np.float32), config, "cpu")
    assert truncated == 1
    assert len(notes) == 1
    assert notes[0]["onset"] == 0
    assert notes[0]["offset"] == 1

    class Empty:
        def generate(self, *args, **kwargs):
            return torch.tensor([[sm_tokenizer.EOS]])

    notes, truncated = pipeline.predict_audio(Empty(), Frontend(), np.zeros(8, np.float32), config, "cpu")
    assert notes == []
    assert truncated == 0
