"""Generate three aligned toy audio-score pairs from explicit note schedules.

No pretrained model, MIDI conversion or real evaluation data. Run from repo root.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf


def make_demo(output, repetitions=1):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # (start in quarter notes, duration in quarter notes, MIDI pitch number).
    # Pitch numbers here are only frequency labels for the synthetic oscillator.
    scores = {
        "mono": [(0, 1, 60), (1, .5, 62), (1.5, .5, 64), (3, 1, 67), (4, 3, 60)],
        "fixed": [(0, 2, 48), (2, 1, 55), (4, 4, 48), (0, 1, 60), (0, 1, 64),
                  (1, 1, 62), (2, 2, 64), (4, 4, 67)],
        "open": [(0, 1, 60), (1, 1, 64), (1, 1, 60), (2, .5, 67), (2, .5, 64),
                 (2, .5, 60), (2.5, .5, 69), (2.5, .5, 65), (2.5, .5, 62),
                 (3, 1, 67), (3, 1, 64), (4, 4, 60), (4, 4, 64), (4, 4, 67)],
    }
    entries = []
    for name, notes in scores.items():
        sr, seconds_per_quarter = 16000, .25
        signal = np.zeros(sr * 2, dtype=np.float32)
        for start, duration, pitch in notes:
            start = round(start * seconds_per_quarter * sr)
            count = round(duration * seconds_per_quarter * sr)
            t = np.arange(count) / sr
            frequency = 440 * 2 ** ((pitch - 69) / 12)
            envelope = np.minimum(t / .005, 1) * np.minimum((count / sr - t) / .02, 1)
            signal[start:start + count] += (.18 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        sf.write(output / f"{name}.wav", np.tile(signal, repetitions), sr, subtype="PCM_16")
        source = Path(__file__).parent / "kern" / f"{name}.krn"
        if repetitions == 1:
            shutil.copyfile(source, output / f"{name}.krn")
        else:
            lines = source.read_text().splitlines()
            first_bar = next(i for i, line in enumerate(lines) if line.startswith("="))
            result = lines[:first_bar]
            for repeat in range(repetitions):
                for line in lines[first_bar:-2]:
                    if line.startswith("="):
                        line = "\t".join("=" + str(2 * repeat + int(cell[1:])) for cell in line.split("\t"))
                    result.append(line)
            (output / f"{name}.krn").write_text("\n".join(result + lines[-2:]) + "\n")
        entries.append({"id": name, "piece_id": f"toy-{name}", "split": "train", "category": name,
                        "audio": f"{name}.wav", "score": f"{name}.krn", "start": 0, "end": 2 * repetitions})
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    return output / "manifest.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/score-demo")
    parser.add_argument("--repetitions", type=int, default=1, help="Repeat the music; 24 creates 48-second complete pairs")
    args = parser.parse_args()
    print(make_demo(args.output, args.repetitions))
