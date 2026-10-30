"""Select the task before importing the optional legacy MIDI training stack."""
import sys


def dispatch(command):
    args = sys.argv[1:]
    task = "audio_to_midi"
    for i, arg in enumerate(args):
        if arg == "--task":
            if i + 1 == len(args):
                raise SystemExit("--task requires audio_to_midi or audio_to_score")
            task = args[i + 1]
            del args[i:i + 2]
            break
        if arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            del args[i]
            break
    if task not in {"audio_to_midi", "audio_to_score"}:
        raise SystemExit(f"Unknown task: {task}")
    sys.argv[1:] = args
    if task == "audio_to_score":
        from a2s.cli import main
        main(command, args)
        raise SystemExit(0)
