"""Legacy tokenizer, loss, raw checkpoints and AR decoder remain usable."""
from pathlib import Path
import pandas as pd
import pretty_midi
import pytest
import torch
import torch.nn.functional as F

from a2s.runtime import configuration
from model.T5 import Transformer
from data.constants import sm_tokenizer, TOKEN_END, TOKEN_START, TOKEN_PAD

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("notes", [
    [(0., .5, 60), (.5, .5, 62)],
    [(0., .5, 48), (0., .5, 60), (.5, .5, 50), (.5, .5, 62)],
    [(0., 1., 60), (.25, .5, 64), (.5, .25, 67)],
], ids=["mono", "fixed", "open"])
def test_midi_tokenizer_training_checkpoint_and_generation(notes, tmp_path):
    events = []
    for start, length, pitch in notes:
        events += [{"onset_sec": start, "pitch": pitch, "type": "NoteOn", "type_id": 2, "velocity": 80},
                   {"onset_sec": start + length, "pitch": pitch, "type": "NoteOff", "type_id": 1, "velocity": 0}]
    tokens, _ = sm_tokenizer.tokenize_dataframe(pd.DataFrame(events))
    decoded = sm_tokenizer.midi_events_to_notes(sm_tokenizer.detokenize(tokens, offsets_sec=0))
    assert len(decoded) == len(notes)
    for actual, expected in zip(sorted(decoded, key=lambda x: (x["onset"], x["pitch"])), sorted(notes, key=lambda x: (x[0], x[2]))):
        assert (actual["onset"], actual["duration"], actual["pitch"]) == expected
    path = tmp_path / "roundtrip.mid"
    sm_tokenizer.save_midi(decoded, str(path))
    assert len(pretty_midi.PrettyMIDI(str(path)).instruments[0].notes) == len(notes)
    cfg = configuration(ROOT / "config/score_smoke.yaml")["model"]
    cfg.update(task="audio_to_midi", vocab_size=1024)
    model = Transformer(cfg)
    targets = torch.tensor([tokens.tolist() + [TOKEN_END]])
    assert model.bos_token == TOKEN_START and model.pad_token == TOKEN_PAD
    audio_features = torch.randn(1, 128, 512)
    loss = F.cross_entropy(model(audio_features, targets)["decoder_outputs"].flatten(0, 1), targets.flatten())
    loss.backward()
    assert model.encoder.dense.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.step()
    path = tmp_path / "legacy.ckpt"
    torch.save(model.state_dict(), path)  # Legacy raw format stays loadable.
    restored = Transformer(cfg)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    model.eval(); restored.eval()
    assert torch.equal(model.generate(audio_features, 6), restored.generate(audio_features, 6))
