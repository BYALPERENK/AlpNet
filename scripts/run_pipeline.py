"""The whole search in one resumable command, each step a fresh process:

1. generation 0: N0 random networks with the first phase's epoch count (train_random.py), the same
   networks with every later phase's epoch count (--like);
2. once: the stop rules calibrated on each (calibrate_rules.py; its chosen rules and report written into
   out/search.yaml, the config the search keeps) and the generation 0 analyses (compare_epochs.py,
   analyze_gen0.py; val only);
3. the GA (run_search.py --gen0).

The same command continues after a stop, a crash or a reboot: finished steps are skipped (they resume
and find nothing to do), a candidate cut in the middle is trained again. A step that fails is started
again RETRIES times (a CUDA error needs a fresh process), then the pipeline stops. More generations in
out/search.yaml's last phase extend a finished search. Stop: end the process tree (its id is in
out/pipeline.pid). The process asks Windows not to sleep while it runs (no setting is changed).

    python scripts/run_pipeline.py --out runs/main_20261024
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RETRIES = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "search.yaml", help="before calibration")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline.pid").write_text(str(os.getpid()), encoding="utf-8")
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, for this process's lifetime only
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log_f = open(out / "pipeline.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    def run(name, *cmd, required=True):
        for attempt in range(1 + RETRIES):
            log(f"{name}: start" + (f" (try {attempt + 1})" if attempt else ""))
            with open(out / f"{name}.log", "a", encoding="utf-8") as f:
                code = subprocess.call([sys.executable, *map(str, cmd)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            log(f"{name}: exit {code}")
            if code == 0:
                return True
        if required:
            log(f"{name}: failed {1 + RETRIES} times, stopped; the same command continues")
            raise SystemExit(1)
        return False

    cfg_path = out / "search.yaml"
    base = yaml.safe_load((cfg_path if cfg_path.exists() else args.config).read_text(encoding="utf-8"))
    epochs = [p["epochs"] for p in base["phases"]]
    gen0 = [out / f"gen0_{e}ep" for e in epochs]
    run(f"gen0_{epochs[0]}ep", "scripts/train_random.py", "--n", base["n0"], "--epochs", epochs[0], "--out", gen0[0])
    for e, d in zip(epochs[1:], gen0[1:]):
        run(f"gen0_{e}ep", "scripts/train_random.py", "--epochs", e, "--like", gen0[0], "--out", d)

    if not cfg_path.exists():  # written once: the search keeps the config it started with
        for d in gen0:
            run(f"calibrate_{d.name}", "scripts/calibrate_rules.py", d, "--config", args.config)
        for p, d in zip(base["phases"], gen0):
            text = (d / "calibration.md").read_text(encoding="utf-8")
            block = yaml.safe_load(text.split("## Chosen rules")[1].split("```yaml\n")[1].split("```")[0])
            p["rules"], p["calibration"] = block["rules"], block["calibration"]
            log(f"{p['epochs']} epochs: rules {block['rules']}")
        cfg_path.write_text(f"# {args.config.name} with the stop rules calibrated on generation 0 "
                            "(scripts/run_pipeline.py)\n" + yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        run("compare_epochs", "scripts/compare_epochs.py", *gen0, required=False)
        for d in gen0:
            run(f"analyze_{d.name}", "scripts/analyze_gen0.py", d, base["k"], required=False)

    run("search", "scripts/run_search.py", "--out", out / "search", "--config", cfg_path, "--gen0", *gen0)
    log("done")


if __name__ == "__main__":
    main()
