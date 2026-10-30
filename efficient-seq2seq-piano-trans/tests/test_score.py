import copy
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from a2s.tokenizer import KernTokenizer
from a2s.kern import validate, duration
from a2s.data import ScoreDataset, collate, read_manifest
from a2s.runtime import configuration, frontend, features, load_checkpoint, save_checkpoint
from a2s.cli import serialize_prediction, edit_distance
from model.T5 import Transformer
from examples.make_score_demo import make_demo

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,count", [("mono", 1), ("fixed", 2), ("open", 3)])
def test_lossless_valid_score(name, count, tmp_path):
    text = (ROOT / "examples/kern" / (name + ".krn")).read_text()
    tok = KernTokenizer()
    assert tok.decode(tok.encode(text)) == text
    assert validate(text, external=True)["max_spines"] == count
    out = tmp_path / (name + ".krn")
    assert serialize_prediction(tok.encode(text), out)["valid"]
    assert out.read_text() == text


def test_unbounded_spines_and_byte_vocabulary():
    text = "\n".join("\t".join([cell] * 37) for cell in ["**kern", "4c", "*-"]) + "\n"
    text = "!!!OTL: Música 🎹\n" + text
    assert validate(text)["max_spines"] == 37
    tok = KernTokenizer()
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 259


@pytest.mark.parametrize("text", [
    "**kern\n*^\n4c\n*-\n", "**kern\n*v\n4c\n*-\n",
    "**kern\n4c\n", "**kern\n.\n*-\n", "**kern\nxyz\n*-\n",
    "**kern\n4c 8e\n*-\n", "**kern\n*+\n4c\t4e\n*-\t*-\n",
    "**kern\n*^\n2c\t4e\n*v\t*v\n4c\n*-\n",
])
def test_invalid_scores_rejected(text):
    with pytest.raises(ValueError):
        validate(text)


def test_spine_add_exchange_terminate():
    text = "**kern\n4c\n*+\n*\t**kern\n4d\t4E\n*x\t*x\n4F\t4e\n*-\t*\n4f\n*-\n"
    assert validate(text, external=True)["max_spines"] == 2


def test_reciprocals_and_ties():
    from fractions import Fraction
    assert duration("8..c#") == Fraction(7, 8)
    assert duration("3%2c") == Fraction(8, 3)
    assert duration("00C") == 16
    assert duration("16qcc") == 0
    text = "**kern\n*M4/4\n=1\n[1c\n=2\n1c]\n==\n*-\n"
    assert validate(text, external=True)["quarter_length"] == "8"


def test_invalid_or_truncated_prediction_is_not_score(tmp_path):
    tok = KernTokenizer()
    path = tmp_path / "bad.krn"
    ids = tok.encode("**kern\n4c\n*-\n")[:-1]
    assert not serialize_prediction(ids, path)["valid"]
    assert not path.exists()
    with pytest.raises(ValueError):
        tok.decode([3, 0, 2])
    with pytest.raises(ValueError):
        tok.decode([2, 3])


def test_data_training_checkpoint_and_cached_inference(tmp_path):
    manifest = make_demo(tmp_path / "demo")
    cfg = configuration(ROOT / "config/score_smoke.yaml")
    data = ScoreDataset(manifest, cfg["data"], external=True)
    audio, targets, _ = collate([data[i] for i in range(3)])
    x = features(frontend(cfg, "cpu"), audio)
    assert x.shape == (3, 128, 512)
    model = Transformer(cfg["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    logits = model(x, targets)["decoder_outputs"]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)
    loss.backward()
    assert model.encoder.dense.weight.grad.abs().sum() > 0
    assert model.decoder.token_embed.embedding.grad.abs().sum() > 0
    optimizer.step()
    checkpoint = tmp_path / "score.pt"
    save_checkpoint(checkpoint, model, optimizer, cfg, 1, data.fingerprint())
    restored, metadata = load_checkpoint(checkpoint)
    assert metadata["step"] == 1
    model.eval(); restored.eval()
    assert torch.equal(model.generate(x, 8), restored.generate(x, 8))
    # Compare cached next-token logits to teacher forcing for all three cases.
    with torch.no_grad():
        expected = model(x, targets[:, :80], enable_dropout=False)["decoder_outputs"]
        model.decoder.initialize_decoder_cache()
        encoded = model.encode(x, enable_dropout=False)
        shifted = model._shift_right(targets[:, :80])
        for i in range(80):
            step = model.decode(encoded, encoded, shifted[:, i:i+1], torch.ones(3, 1, dtype=torch.long),
                                decoder_positions=torch.full((3, 1), i), enable_dropout=False, decode=True)["decoder_outputs"]
            active = targets[:, i] != KernTokenizer.PAD
            torch.testing.assert_close(step[active, 0], expected[active, i], atol=2e-5, rtol=2e-5)


def test_checkpoint_task_and_vocab_rejection(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"decoder.dense.weight": torch.zeros(2, 2)}, path)
    with pytest.raises(ValueError, match="MIDI"):
        load_checkpoint(path)
    torch.save({"task": "audio_to_score", "format_version": 1, "tokenizer": {}}, path)
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(path)


def test_manifest_leakage_and_no_truncation(tmp_path):
    path = make_demo(tmp_path)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    entries.append({**entries[0], "id": "leak", "split": "test"})
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(path)
    path = make_demo(tmp_path)
    cfg = configuration(ROOT / "config/score_smoke.yaml")
    cfg["data"]["max_token_length"] = 5
    with pytest.raises(ValueError, match="tokens exceed"):
        ScoreDataset(path, cfg["data"])
    cfg["data"]["max_token_length"] = 512
    cfg["data"]["n_frames"] = 4
    with pytest.raises(ValueError, match="interval too long"):
        ScoreDataset(path, cfg["data"])


def test_edit_distance():
    assert edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert edit_distance([], [1, 2]) == 2


def test_audio_alignment_and_resampling(tmp_path):
    import numpy as np
    import soundfile as sf
    from a2s.data import load_audio
    path = make_demo(tmp_path)
    cfg = configuration(ROOT / "config/score_smoke.yaml")["data"]
    audio, sr = sf.read(tmp_path / "open.wav", dtype="float32")
    sf.write(tmp_path / "shifted.wav", np.concatenate([np.zeros(sr), audio]), sr, subtype="FLOAT")
    direct = load_audio(tmp_path / "open.wav", 0, 2, cfg)
    shifted = load_audio(tmp_path / "shifted.wav", 1, 3, cfg)
    torch.testing.assert_close(direct, shifted, atol=0, rtol=0)
    # Stereo, different sample rate, and an interval shorter than n_frames.
    stereo = np.zeros((22050, 2), dtype="float32")
    sf.write(tmp_path / "stereo.wav", stereo, 22050)
    result = load_audio(tmp_path / "stereo.wav", 0, 1, cfg)
    assert result.shape == (128 * 320,) and torch.count_nonzero(result) == 0


def test_resume_matches_uninterrupted_training(tmp_path):
    from argparse import Namespace
    from a2s.cli import train
    manifest = make_demo(tmp_path / "demo")
    options = dict(manifest=str(manifest), config=str(ROOT / "config/score_smoke.yaml"), device="cpu")
    train(Namespace(**options, output=str(tmp_path / "full"), resume=None, steps=4))
    train(Namespace(**options, output=str(tmp_path / "split"), resume=None, steps=2))
    resume = tmp_path / "split/latest.pt"
    train(Namespace(**options, output=str(tmp_path / "split"), resume=str(resume), steps=4))
    full, a = load_checkpoint(tmp_path / "full/latest.pt")
    resumed, b = load_checkpoint(resume)
    assert a["step"] == b["step"] == 4
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
