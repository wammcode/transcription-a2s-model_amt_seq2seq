"""Task-selected preparation; MIDI preserves the original MAESTRO preparation."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    from task_dispatch import dispatch
    dispatch("prepare")
    if len(sys.argv) > 1:
        raise SystemExit("MIDI preparation uses dataset/maestro-v3.0.0; no extra arguments")
    for script in ("audio_to_h5_MAESTRO", "midi_2_performance_tsv_MAESTRO"):
        subprocess.run([sys.executable, "-m", "data." + script],
                       cwd=Path(__file__).resolve().parent, check=True)
