import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from a2s.data import ScoreDataset, FeatureDataset, LengthBatchSampler, load_audio, collate_features
from a2s.runtime import configuration, frontend, features, load_checkpoint, save_checkpoint
from a2s.tokenizer import KernTokenizer, KernFieldTokenizer, tokenizer_from_metadata
from a2s.kern import validate
from a2s.training import train
from model.Attention import Multi_Head_Attention
from model.Layers import FixedEmbed, sinusoidal
from model.T5 import Transformer
from examples.make_score_demo import make_demo

ROOT = Path(__file__).resolve().parents[1]


def small_long():
    cfg = configuration(ROOT / "config/score_smoke.yaml")
    cfg["data"].update(variable_length=True, max_audio_seconds=120, stft_backend="fft")
    cfg["model"].update(attention_chunk_size=16, gradient_checkpointing=True)
    return cfg


def test_compact_tokenizer_lossless_and_fallback():
    texts = [(ROOT / "examples/kern" / (n + ".krn")).read_text() for n in ["mono", "fixed", "open"]]
    tok = KernFieldTokenizer.fit(texts)
    assert sum(len(tok.encode(t)) for t in texts) < sum(len(KernTokenizer().encode(t)) for t in texts)
    for text in texts + ["!!!OTL: Música inédita 🎷\n**kern\n*staff128\n3%7cccc###\n*-\n"]:
        assert tok.decode(tok.encode(text)) == text
    assert tokenizer_from_metadata(tok.metadata()).metadata() == tok.metadata()
    with pytest.raises(ValueError):
        tokenizer_from_metadata({**tok.metadata(), "sha256": "wrong"})
    assert "3%7cccc###" not in tok.lookup  # Held-out symbols never enter fitted vocabulary.


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 8])
def test_block_attention_matches_dense_outputs_and_gradients(causal, window):
    layer = Multi_Head_Attention(2, 8, window_size=window, is_causal=causal).double()
    layer.chunk_size = 7
    q = torch.randn(2, 37, 2, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 37, 2, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 37, 2, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.arange(37)[None] < torch.tensor([37, 23])[:, None]
    distance = torch.arange(37)[:, None] - torch.arange(37)[None]
    allowed = valid[:, None, None, :].expand(-1, 1, 37, -1).clone()
    if causal:
        allowed &= distance >= 0
    if window is not None:
        allowed &= distance <= window if causal else distance.abs() <= window // 2
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed).transpose(1, 2)
    actual = layer.block_attention(q, k, v, valid[:, None, None, :], 0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual.sum(), (q, k, v), retain_graph=True)
    b = torch.autograd.grad(expected.sum(), (q, k, v))
    for lhs, rhs in zip(a, b):
        torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_attention_never_materializes_whole_local_matrix(monkeypatch):
    original = F.scaled_dot_product_attention
    sizes = []
    def checked(q, k, v, **kwargs):
        sizes.append((q.size(-2), k.size(-2)))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", checked)
    layer = Multi_Head_Attention(2, 8, window_size=64).eval()
    layer.chunk_size = 128
    layer(torch.randn(1, 2500, 16), torch.randn(1, 2500, 16))
    assert max(q for q, _ in sizes) <= 128
    assert max(k for _, k in sizes) <= 192


def test_dynamic_positions_keep_original_function_and_state_shape():
    embed = FixedEmbed(32, max_length=16)
    positions = torch.arange(2300)[None]
    expected = sinusoidal((2300, 32))
    torch.testing.assert_close(embed(positions)[0], expected, atol=0, rtol=0)
    assert embed.state_dict()["embedding"].shape == (16, 32)


def test_fft_frontend_equals_original_convolution():
    cfg = configuration(ROOT / "config/score_smoke.yaml")
    a = frontend(cfg, "cpu")
    cfg["data"]["stft_backend"] = "fft"
    b = frontend(cfg, "cpu")
    signal = torch.rand(2, 16000, generator=torch.Generator().manual_seed(12)) * .6 - .3
    torch.testing.assert_close(features(a, signal), features(b, signal), atol=4e-4, rtol=5e-5)


def test_variable_padding_does_not_change_predictions_or_pooling():
    cfg = small_long()
    model = Transformer(cfg["model"]).eval()
    x = torch.randn(1, 35, 512)
    target = torch.randint(3, 259, (1, 27))
    alone = model(x, target)["decoder_outputs"]
    padded = torch.cat([x, torch.randn(1, 22, 512) * 100], dim=1)
    together = model(padded, target, encoder_lengths=torch.tensor([35]))["decoder_outputs"]
    torch.testing.assert_close(alone, together, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(model.generate(x, 9), model.generate(padded, 9, encoder_lengths=torch.tensor([35])))


def test_complete_audio_over_40_seconds_forward_backward_and_generation(tmp_path):
    cfg = small_long()
    sr = cfg["data"]["sample_rate"]
    audio = np.sin(2 * np.pi * 440 * np.arange(sr * 45) / sr).astype("float32") * .2
    sf.write(tmp_path / "long.wav", audio, sr)
    wave = load_audio(tmp_path / "long.wav", 0, 45, cfg["data"])
    assert wave.shape == (45 * sr,)
    acoustic = features(frontend(cfg, "cpu"), wave[None])
    assert acoustic.shape == (1, 2250, 512)
    model = Transformer(cfg["model"])
    labels = torch.tensor([KernTokenizer().encode("**kern\n4c\n*-\n")])
    loss = F.cross_entropy(model(acoustic, labels)["decoder_outputs"].flatten(0, 1), labels.flatten())
    loss.backward()
    assert model.encoder.dense.weight.grad.norm() > 0
    result = model.eval().generate(acoustic, 75)
    assert result.size(1) <= 75
    for layer in model.decoder.decoder_layers:
        assert layer.self_attention.cached_key.size(1) <= 65
        assert layer.encoder_decoder_attention.static_key.size(1) <= 2250


def test_cache_and_length_buckets(tmp_path):
    cfg = small_long()
    data = ScoreDataset(make_demo(tmp_path / "demo"), cfg["data"])
    cached = FeatureDataset(data, tmp_path / "cache")
    first = cached[0][0]
    assert cached.cache_path(0).exists()
    cached.extractor = None
    torch.testing.assert_close(first, cached[0][0], atol=0, rtol=0)
    settings = dict(seed=7, batch=3, bucket_size=8, max_batch_frames=220)
    sampler = LengthBatchSampler(data, settings)
    assert sorted(i for batch in sampler for i in batch) == [0, 1, 2]
    assert all(len(batch) <= 2 for batch in sampler)
    assert sampler.batches == LengthBatchSampler(data, settings).batches
    assert list(LengthBatchSampler(data, settings, start_batch=1)) == sampler.batches[1:]
    settings["max_batch_frames"] = 1
    with pytest.raises(ValueError, match="budget"):
        LengthBatchSampler(data, settings)


def test_saxophone_whole_rest_and_excluded_dynamics():
    assert validate("**kern\n*M4/4\nrr2\n4c\n*-\n", external=True)["quarter_length"] == "3"
    with pytest.raises(ValueError, match="exclusive"):
        validate("**kern\t**dynam\n4c\tp\n*-\t*-\n")


def test_accumulation_and_sampler_resume_are_exact(tmp_path):
    manifest = make_demo(tmp_path / "demo")
    cfg = small_long()
    cfg["training"].update(batch=1, bucket_size=8, gradient_accumulation=2, num_workers=0,
                           precision="fp32", warmup_steps=2, cosine_decay=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    options = dict(manifest=str(manifest), config=str(config), device="cpu", tokenizer=None, set=[], cache_dir=str(tmp_path / "features"))
    train(SimpleNamespace(**options, output=str(tmp_path / "full"), resume=None, steps=3))
    train(SimpleNamespace(**options, output=str(tmp_path / "split"), resume=None, steps=1))
    train(SimpleNamespace(**options, output=str(tmp_path / "split"), resume=str(tmp_path / "split/latest.pt"), steps=3))
    full, a = load_checkpoint(tmp_path / "full/latest.pt")
    resumed, b = load_checkpoint(tmp_path / "split/latest.pt")
    assert a["training_state"] == b["training_state"]
    assert a["scheduler"] == b["scheduler"]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)


def test_three_long_polyphonies_compact_training_and_checkpoint(tmp_path):
    manifest = make_demo(tmp_path / "long", repetitions=24)
    cfg = small_long()
    cfg["model"]["max_decode_length"] = cfg["data"]["max_token_length"] = 2048
    tok = KernFieldTokenizer.fit(p.read_text() for p in (tmp_path / "long").glob("*.krn"))
    cfg["tokenizer"] = tok.metadata()
    cfg["model"]["vocab_size"] = tok.vocab_size
    dataset = ScoreDataset(manifest, cfg["data"], external=True, tokenizer=tok)
    assert [r["max_spines"] for r in dataset.reports] == [1, 2, 3]
    assert all(r["quarter_length"] == "192" for r in dataset.reports)
    cached = FeatureDataset(dataset)
    x, y, _, lengths = collate_features([cached[i] for i in range(3)])
    assert x.shape == (3, 2400, 512)
    model = Transformer(cfg["model"])
    optimizer = torch.optim.AdamW(model.parameters())
    loss = F.cross_entropy(model(x, y, encoder_lengths=lengths)["decoder_outputs"].flatten(0, 1), y.flatten(), ignore_index=0)
    loss.backward()
    assert model.encoder.dense.weight.grad.norm() > 0
    optimizer.step()
    path = tmp_path / "long.pt"
    save_checkpoint(path, model, optimizer, cfg, 1, dataset.fingerprint())
    restored, ckpt = load_checkpoint(path)
    assert ckpt["tokenizer"] == tok.metadata()
    torch.testing.assert_close(model.eval().generate(x, 8), restored.eval().generate(x, 8))


def test_early_stopping_saves_latest_and_best(tmp_path, monkeypatch):
    manifest = make_demo(tmp_path / "demo")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[-1]["split"] = "validation"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg = small_long()
    cfg["training"].update(save_every=1, early_stopping_patience=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr("a2s.training.validate_loss", lambda *args: 1.0)
    train(SimpleNamespace(config=str(config), manifest=str(manifest), resume=None, output=str(tmp_path / "run"), steps=10, device="cpu"))
    _, latest = load_checkpoint(tmp_path / "run/latest.pt")
    _, best = load_checkpoint(tmp_path / "run/best.pt")
    assert latest["step"] == 2 and best["step"] == 1
    assert latest["training_state"]["bad_validation_checks"] == 1


def test_evaluation_compares_scores_not_tokenization_choices(tmp_path, monkeypatch):
    from a2s.cli import evaluate
    manifest = make_demo(tmp_path / "demo")
    texts = [(tmp_path / "demo" / (name + ".krn")).read_text() for name in ["mono", "fixed", "open"]]
    tok = KernFieldTokenizer.fit(texts)
    cfg = small_long()
    cfg["tokenizer"], cfg["model"]["vocab_size"] = tok.metadata(), tok.vocab_size
    # The same score is valid as learned fields OR its fallback bytes.
    predictions = iter(torch.tensor([KernTokenizer().encode(text)]) for text in texts)
    model = SimpleNamespace(generate=lambda *args, **kwargs: next(predictions), eval=lambda: None)
    monkeypatch.setattr("a2s.cli.load_checkpoint", lambda *args: (model, {"config": cfg, "tokenizer": tok.metadata(), "step": 0}))
    evaluate(SimpleNamespace(checkpoint="unused", device="cpu", manifest=str(manifest), split="train", output=str(tmp_path / "eval")))
    report = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert report["exact_match_rate"] == 1 and report["symbol_error_rate"] == 0
    assert not any(row["token_exact_match"] for row in report["examples"])
