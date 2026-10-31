"""Copy what the analyses of a finished search need into results/, small enough for the repository.

results/search: results.jsonl.gz (every candidate, val scores), generations.jsonl, ga_state.json, search.yaml and
test_sealed.jsonl.gz (each candidate's sealed test scores at its best val macro-F1 epoch, the confusion matrix
left out). analyze_search.py, analyze_test.py and train_final.py read this folder like the search's own.
results/final: runs.jsonl.gz, the summary.json of every final and baseline run (per seed; genotypes are in
networks.json), the numbers behind summary.md's mean ± std.

    python scripts/export_results.py runs/main_20261024 runs/final_20261027 --baselines 20261021-1647
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpnet.search.ga import read_jsonl  # noqa: E402


def write_gz(path: Path, lines: list[dict] | bytes):
    data = lines if isinstance(lines, bytes) else "".join(json.dumps(r) + "\n" for r in lines).encode("utf-8")
    if path.exists() and gzip.decompress(path.read_bytes()) == data:
        return  # same content: the file is left as it is
    path.write_bytes(gzip.compress(data, mtime=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main", type=Path, help="the run_pipeline.py folder")
    ap.add_argument("final", type=Path, help="the train_final.py folder")
    ap.add_argument("--baselines", required=True, help="folder prefix of the baseline runs in runs/")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    search, out_s, out_f = args.main / "search", args.out / "search", args.out / "final"
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    recs = read_jsonl(search / "results.jsonl", repair=False)
    write_gz(out_s / "results.jsonl.gz", (search / "results.jsonl").read_bytes())  # as the search wrote it
    for src, name in ((search / "generations.jsonl", "generations.jsonl"), (search / "ga_state.json", "ga_state.json"),
                      (args.main / "search.yaml", "search.yaml")):
        shutil.copyfile(src, out_s / name)
    sealed = []
    for r in recs:
        p = ROOT / r["run_dir"] / "test_sealed.json"
        if p.exists():
            test = json.loads(p.read_text(encoding="utf-8"))["checkpoints"]["macro_f1"]["test"]
            sealed.append({"run_dir": r["run_dir"], "hash": r["hash"], "epochs": r["epochs"],
                           "test": {k: v for k, v in test.items() if k != "confusion"}})
    write_gz(out_s / "test_sealed.jsonl.gz", sealed)

    runs = []
    for p in sorted(args.final.glob("*/summary.json")) + sorted((ROOT / "runs").glob(f"{args.baselines}*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        runs.append({"run": p.parent.name} | {k: v for k, v in s.items() if k not in ("genotype", "canonical")})
    write_gz(out_f / "runs.jsonl.gz", runs)
    print(f"{len(recs)} candidates, {len(sealed)} sealed test scores -> {out_s}; {len(runs)} runs -> {out_f}")


if __name__ == "__main__":
    main()
