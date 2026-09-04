from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent

FILES = [
    "tetris_core.py",
    "tetris_placement.py",
    "gym_executor.py",
    "teacher.py",
    "v8_successor.py",
    "ai/state_encoder.py",
    "v8_4_observable.py",
    "v8_4_replay.py",
    "ai/observable_q_network.py",
]

missing = [name for name in FILES if not (ROOT / name).is_file()]
if missing:
    print("Missing files:")
    for name in missing:
        print(" -", name)
    raise SystemExit(2)

out = ROOT / "v8_8_bridge_sources.zip"
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(ROOT / name, arcname=name)

print("Created:", out)
print("Files:", len(FILES))
print("Upload this ZIP if the V8.8 parity gate reports any mismatch,")
print("or to let ChatGPT wire the verified JAX backend into the production trainer.")
