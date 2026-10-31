"""GA tests with a fake trainer (CPU only, decoding is real). Runs with pytest or directly:

    python tests/test_ga.py
"""

import gzip
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.models.decoder import MAX_MACS, MAX_PARAMS, decode  # noqa: E402
from alpnet.search.fitness import fitness  # noqa: E402
from alpnet.search.ga import (MAX_ERRORS, GAConfig, Interrupted, Search, active_entropy, cache_key,  # noqa: E402
                              eligible, make_rules, pareto_front, read_jsonl)
from alpnet.search.space import SLOTS, sample  # noqa: E402
from alpnet.train.stopping import replay  # noqa: E402

RULES10 = {"not_learning": {"epoch": 3}, "floor": {5: 0.3}, "percentile": {3: 0.2, 5: 0.3},
           "reach_elite": {"check": [5, 8], "q": 0.9}, "patience": None}
RULES6 = {"not_learning": {"epoch": 3}, "floor": None, "percentile": {3: 0.2}, "reach_elite": {"check": [4], "q": 0.9},
          "patience": None}
RULES40 = {"not_learning": {"epoch": 3}, "floor": {10: 0.60}, "percentile": {5: 0.2, 10: 0.3, 20: 0.5},
           "reach_elite": {"check": [5, 10, 20, 30], "q": 0.9}, "patience": {"patience": 5, "start": 30}}
SMALL = dict(n0=8, k=4, c=4, m=2, seed=1, phases=[{"epochs": 10, "generations": 4, "rules": RULES10}])
TWO = dict(n0=12, k=4, c=4, m=2, seed=1, phases=[{"epochs": 6, "generations": 2, "rules": RULES6},
                                                 {"epochs": 10, "generations": 2, "rules": RULES10}])
KEY = cache_key({"epochs": 10, "lr": 0.002}, "toy@0123", 0, 5)


def fake_trainer(calls: list, key=None, fail=lambda n: False, rules_seen=None):
    """Final F1 grows with the number of active slots plus hash noise, 5% lower with fewer than 10 epochs;
    ~5% of candidates "OOM"; fail(call number) -> an error instead."""
    def evaluate(idx, g, ph, rules, gen_dir, epochs):
        n = len(calls)
        calls.append((ph.hash, epochs))
        if rules_seen is not None:
            rules_seen.append((epochs, rules))
        row = {"idx": idx, "hash": ph.hash, "params": ph.params, "macs": ph.macs, "genotype": g, "epochs": epochs}
        if key is not None:
            row["key"] = key
        if fail(n):
            return row | {"status": "error: RuntimeError: boom", "total_s": 0.1}
        noise = int(ph.hash[:6], 16) / 0xFFFFFF
        if noise < 0.05:
            return row | {"status": "oom (>= 20 GB)", "total_s": 0.1}
        final = (0.4 + 0.08 * sum(g[f"s{r}.active"] for r in SLOTS) + 0.1 * noise) * (0.95 if epochs < 10 else 1)
        rows = [{"epoch": e, "val_macro_f1": final * (e / epochs) ** 0.5, "train_loss": 2.0, "train_acc": 0.3}
                for e in range(1, epochs + 1)]
        e, stop = replay(rules, rows)
        best = max(r["val_macro_f1"] for r in rows[:e])
        return row | {"status": f"pruned: {stop.reason}" if stop and stop.prune else "completed",
                      "stop": stop and stop.reason, "epochs_done": e, "curve": [r["val_macro_f1"] for r in rows[:e]],
                      "val": {"macro_f1": best}, "fitness": fitness(best, ph.params, ph.macs), "total_s": 1.0}
    return evaluate


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def quiet(*a, **k):
    return Search(*a, **k, log=lambda msg: None)


def interrupted(cfg, out, evaluate, stop_after, **k):
    try:
        quiet(cfg, out, evaluate, stop_after=stop_after, **k).run()
        raise AssertionError("not interrupted")
    except Interrupted:
        pass


def test_config():
    cfg = GAConfig(phases=[{"epochs": 20, "generations": 10, "rules": {}}, {"epochs": 40, "generations": 10, "rules": {}}])
    assert cfg.generations == 20 and cfg.epoch_counts == [20, 40] and cfg.ranges() == [(1, 10, 20), (11, 20, 40)]
    assert [cfg.phase(g)["epochs"] for g in (0, 1, 10, 11, 20)] == [20, 20, 20, 40, 40]
    for bad in ([], [{"epochs": 20, "start": 1, "rules": {}}], [{"epochs": 20, "generations": 0, "rules": {}}]):
        try:
            GAConfig(phases=bad)
            raise AssertionError(f"accepted {bad}")
        except AssertionError as e:
            assert "accepted" not in str(e)


def test_search_runs():
    with tempfile.TemporaryDirectory() as d:
        calls = []
        quiet(GAConfig(**SMALL), Path(d), fake_trainer(calls)).run()
        recs = lines(Path(d) / "results.jsonl")
        assert len(recs) == len(calls) and len({r["hash"] for r in recs}) == len(recs)  # nothing trained twice
        assert all(r["params"] <= MAX_PARAMS and r["macs"] <= MAX_MACS and r["epochs"] == 10 for r in recs)
        assert [r["eval"] for r in recs] == list(range(1, len(recs) + 1))
        assert sum(eligible(r) for r in recs if r["generation"] == 0) == SMALL["n0"]
        for g in range(1, 5):
            kids = [r for r in recs if r["generation"] == g]
            assert len(kids) == SMALL["c"] + SMALL["m"]
            for r in kids:
                assert all(p["generation"] < g for p in recs if p["hash"] in r["parents"])
        st = json.loads((Path(d) / "ga_state.json").read_text())
        by_hash = {r["hash"]: r for r in recs}
        assert st["generation"] == 4 and len(st["population"]) <= SMALL["k"] + SMALL["c"] + SMALL["m"]
        assert all(eligible(by_hash[p["hash"]]) for p in st["population"])
        assert json.loads((Path(d) / "ga_state.prev.json").read_text())["generation"] == 3
        assert [x["generation"] for x in lines(Path(d) / "generations.jsonl")] == list(range(5))
        assert any(r["status"].startswith("pruned") for r in recs) and any(not eligible(r) for r in recs)


def test_restart_continues_identically():
    """Interrupted at the start of the 10-epoch phase (in the middle of re-training the best networks)."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        full_calls, calls = [], []
        quiet(GAConfig(**TWO), Path(a), fake_trainer(full_calls)).run()
        ra = lines(Path(a) / "results.jsonl")
        cut = next(i for i, r in enumerate(ra) if r["origin"] == "reeval" and r["generation"] == 3) + 2
        interrupted(GAConfig(**TWO), Path(b), fake_trainer(calls), cut)
        quiet(GAConfig(**TWO), Path(b), fake_trainer(calls)).run()
        rb = lines(Path(b) / "results.jsonl")
        assert [(r["hash"], r["epochs"]) for r in ra] == [(r["hash"], r["epochs"]) for r in rb]
        assert calls == full_calls  # nothing retrained
        sa, sb = (json.loads((Path(x) / "ga_state.json").read_text()) for x in (a, b))
        assert sa == sb
        assert lines(Path(a) / "generations.jsonl") == lines(Path(b) / "generations.jsonl")


def make_gen0(folder: Path, nets: list, epochs: int, ev):
    """A train_random.py run of `nets` with `epochs` epochs."""
    folder.mkdir()
    rows = [ev(i + 1, g, ph, [], folder, epochs) for i, (g, ph) in enumerate(nets)]
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_phases_and_switch():
    """Generation 0 imported at 6 and 10 epochs; 6 epochs in generations 1-2, 10 in 3-4."""
    with tempfile.TemporaryDirectory() as d:
        src6, src10, out = Path(d) / "six", Path(d) / "ten", Path(d) / "search"
        ev, rng, nets = fake_trainer([], key=KEY), random.Random(5), []
        while len(nets) < 14:
            g = sample(rng)
            ph = decode(g)
            if ph.within_budget() and ph.hash not in {p.hash for _, p in nets}:
                nets.append((g, ph))
        rows6, rows10 = make_gen0(src6, nets, 6, ev), make_gen0(src10, nets, 10, ev)
        assert sum(map(eligible, rows6)) >= TWO["n0"]
        with open(src6 / "results.jsonl", "a", encoding="utf-8") as f:  # failed with an error: not imported
            f.write(json.dumps({"idx": 99, "hash": "0" * 16, "status": "error: CUDA", "epochs": 6, "key": KEY}) + "\n")
        part10 = Path(d) / "ten_part"  # one network of the 6-epoch population missing at 10 epochs
        missing = next(r for r in rows6 if eligible(r))["hash"]
        make_gen0(part10, [(g, ph) for g, ph in nets if ph.hash != missing], 10, ev)
        failed10 = Path(d) / "ten_failed"  # every network there, but too few completed: no reference curves
        failed10.mkdir()
        (failed10 / "results.jsonl").write_text("".join(json.dumps(r | {"status": "failed: loss nan at epoch 2"}) + "\n"
                                                        for r in rows10), encoding="utf-8")
        for srcs, key in (((src6, src10), KEY | {"decoder_version": 99}), ((src6,), KEY), ((src6, part10), KEY),
                          ((src6, failed10), KEY)):
            try:  # another cache key; an epoch count of a phase missing; not the same networks; too few curves
                quiet(GAConfig(**TWO), Path(d) / f"other{len(srcs)}", ev, key=key).import_generation0(srcs)
                raise AssertionError("import not refused")
            except ValueError:
                pass
        # an import cut short before its state was saved (results.jsonl written, no ga_state.json) is redone
        first = quiet(GAConfig(**TWO), out, ev, key=KEY)
        first.import_generation0((src6, src10))
        (out / "ga_state.json").unlink()
        (out / "generations.jsonl").unlink()
        rules_seen = []
        s = quiet(GAConfig(**TWO), out, fake_trainer([], key=KEY, rules_seen=rules_seen), key=KEY)
        s.import_generation0((src6, src10))
        assert len(lines(out / "results.jsonl")) == len(first.history) == 28 and "0" * 16 not in s.genotypes
        assert {p["hash"] for p in s.population} == {r["hash"] for r in rows6 if eligible(r)}
        for e, rows in ((6, rows6), (10, rows10)):  # the stop rules' reference: generation 0's curves
            assert s.reference(e) == [r["curve"] for r in rows if r["status"] == "completed"]
        s.run()
        recs = lines(out / "results.jsonl")
        assert Counter((r["hash"], r["epochs"]) for r in recs).most_common(1)[0][1] == 1
        for g in (1, 2):  # the 6-epoch phase: children only
            assert {r["epochs"] for r in recs if r["generation"] == g} == {6}
            assert not [r for r in recs if r["generation"] == g and r["origin"] == "reeval"]
        # A's thresholds are the same for every child of a phase (fixed reference, not the growing history)
        for e in (6, 10):
            th = {tuple(sorted(r.thresholds.items())) for ep, rules in rules_seen if ep == e and rules
                  for r in rules if getattr(r, "name", "") == "percentile"}
            assert len(th) == 1, (e, th)
        g3 = [r for r in recs if r["generation"] == 3]
        reeval = [r for r in g3 if r["origin"] == "reeval"]
        assert reeval and all(r["epochs"] == 10 for r in g3)
        assert all(len(r["parents"]) == 1 and r["parents"][0] == r["hash"] for r in reeval)
        assert all(r["hash"] in {x["hash"] for x in recs if 0 < x["generation"] < 3} for r in reeval)  # gen 0 has 10
        # trained again: the elites and the K best of every 6-epoch network (dead elites included)
        done6 = [r for r in recs if r["epochs"] == 6 and eligible(r)]
        best6 = sorted(done6, key=lambda r: -r["fitness"])[:TWO["k"]]
        at10 = {r["hash"] for r in recs if r["epochs"] == 10 and r["generation"] <= 3}
        assert all(r["hash"] in at10 for r in best6 + pareto_front(done6, TWO["k"]))  # and the val F1 / MACs front
        assert len(g3) - len(reeval) == TWO["c"] + TWO["m"]
        assert not [r for r in recs if r["generation"] == 4 and r["origin"] == "reeval"]
        assert all(s.fitness_at(p["hash"], 10) is not None for p in s.population)
        diag = lines(out / "generations.jsonl")
        assert [x["epochs"] for x in diag] == [6, 6, 6, 10, 10] and diag[3]["reevaluated"] == len(reeval)


def test_age_rule():
    with tempfile.TemporaryDirectory() as d:
        s = quiet(GAConfig(**SMALL | {"age_max": 0, "death_rate": 1.0}), Path(d), fake_trainer([]))
        s.run()  # every elite is older than 0 after a generation and dies
        assert all(p["age"] == 0 for p in s.population)
        diag = lines(Path(d) / "generations.jsonl")
        assert all(len(x["killed_by_age"]) == x["elites"] for x in diag[1:])


def test_known_children_are_redrawn():
    """Crossover copying parent 1 and mutation changing nothing only give known networks: random instead."""
    with tempfile.TemporaryDirectory() as d:
        cfg = GAConfig(**SMALL | {"p1": 1.0, "p2": 0.0, "qm": 0.0, "max_tries": 3})
        s = quiet(cfg, Path(d), fake_trainer([]))
        s.run()
        later = [r for r in s.history if r["generation"] > 0]
        assert later and all(r["origin"] == "random" for r in later)
        assert len({r["hash"] for r in s.history}) == len(s.history)


def test_restart_checks():
    """A restart refuses another config and records of another cache key; a line cut mid-write is dropped
    and that candidate is trained again."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as d:
        full_calls, calls = [], []
        quiet(GAConfig(**SMALL), Path(a), fake_trainer(full_calls), key=KEY).run()
        stop = SMALL["n0"] + 3  # in generation 1, after its third child
        interrupted(GAConfig(**SMALL), Path(d), fake_trainer(calls), stop, key=KEY)
        res = Path(d) / "results.jsonl"
        assert all(r["key"] == KEY for r in lines(res))
        for cfg, key in ((GAConfig(**SMALL | {"c": 5}), KEY), (GAConfig(**SMALL), KEY | {"seed": 1})):
            try:
                quiet(cfg, Path(d), fake_trainer([]), key=key)
                raise AssertionError(f"not refused: {cfg}, {key}")
            except ValueError:
                pass
        text = res.read_text(encoding="utf-8")
        res.write_text(text[:-30], encoding="utf-8")  # the last record cut in the middle
        quiet(GAConfig(**SMALL), Path(d), fake_trainer(calls), key=KEY).run()
        assert [(r["hash"], r["eval"]) for r in lines(Path(a) / "results.jsonl")] == [(r["hash"], r["eval"]) for r in lines(res)]
        assert calls == full_calls[:stop] + full_calls[stop - 1:]  # only the cut one trained again


def test_crash_recovery():
    """After a power cut: an unreadable ga_state.json (kept aside as ga_state.bad.json), or one whose
    generation lost records in results.jsonl (population members, or just one record), falls back to the
    previous state (that generation is done again); a diagnostics line of an unfinished generation is
    dropped."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        full_calls = []
        quiet(GAConfig(**SMALL), Path(a), fake_trainer(full_calls)).run()
        la = lines(Path(a) / "results.jsonl")
        ra = [(r["hash"], r["eval"]) for r in la]
        stop = next(i for i, r in enumerate(la) if r["generation"] == 3) + 1  # generation 2 finished, 1 child of 3
        calls = []
        interrupted(GAConfig(**SMALL), Path(b), fake_trainer(calls), stop)
        (Path(b) / "ga_state.json").write_bytes(b"\0" * 64)  # renamed, but the data never reached the disk
        quiet(GAConfig(**SMALL), Path(b), fake_trainer(calls)).run()
        assert [(r["hash"], r["eval"]) for r in lines(Path(b) / "results.jsonl")] == ra and calls == full_calls
        assert lines(Path(a) / "generations.jsonl") == lines(Path(b) / "generations.jsonl")
        assert (Path(b) / "ga_state.bad.json").read_bytes() == b"\0" * 64  # kept aside, not made the backup
        assert json.loads((Path(b) / "ga_state.prev.json").read_text())["generation"] == 3

        for lost in ("all", "one"):  # generation 2's records lost; or only one not in the population
            with tempfile.TemporaryDirectory() as c:
                interrupted(GAConfig(**SMALL), Path(c), fake_trainer([]), stop)
                res = Path(c) / "results.jsonl"
                pop = {p["hash"] for p in json.loads((Path(c) / "ga_state.json").read_text())["population"]}
                rs = lines(res)
                gone = [r for r in rs if r["generation"] == 2 and (lost == "all" or r["hash"] not in pop)]
                gone = gone if lost == "all" else gone[:1]
                assert gone
                res.write_text("".join(json.dumps(r) + "\n" for r in rs if r not in gone), encoding="utf-8")
                quiet(GAConfig(**SMALL), Path(c), fake_trainer([])).run()
                assert sorted(((r["hash"], r["eval"]) for r in lines(res)), key=lambda t: t[1]) == ra, lost
                assert lines(Path(a) / "generations.jsonl") == lines(Path(c) / "generations.jsonl")


def test_extend():
    """More generations in the last phase continue a finished run where it stopped; other changes are refused."""
    longer = SMALL | {"phases": [SMALL["phases"][0] | {"generations": 6}]}
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        quiet(GAConfig(**longer), Path(a), fake_trainer([])).run()
        quiet(GAConfig(**SMALL), Path(b), fake_trainer([])).run()
        calls = []
        quiet(GAConfig(**longer), Path(b), fake_trainer(calls)).run()
        assert [(r["hash"], r["eval"]) for r in lines(Path(a) / "results.jsonl")] == \
               [(r["hash"], r["eval"]) for r in lines(Path(b) / "results.jsonl")]
        assert {r["generation"] for r in lines(Path(b) / "results.jsonl")} == set(range(7))
        assert calls and all(h in {r["hash"] for r in lines(Path(b) / "results.jsonl") if r["generation"] > 4}
                             for h, _ in calls)
        for other in (SMALL["phases"][0] | {"generations": 3}, SMALL["phases"][0] | {"generations": 8, "epochs": 9}):
            try:
                quiet(GAConfig(**SMALL | {"phases": [other]}), Path(b), fake_trainer([]))
                raise AssertionError(f"not refused: {other}")
            except ValueError:
                pass


def test_errors():
    """A training failed with an error is trained again at a restart; MAX_ERRORS in a row stop the search."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        quiet(GAConfig(**SMALL), Path(a), fake_trainer([])).run()
        la = lines(Path(a) / "results.jsonl")
        ra = [(r["hash"], r["eval"]) for r in la]
        gen1 = next(i for i, r in enumerate(la) if r["generation"] == 1)
        first = gen1 + 1  # the second child of generation 1 fails once; the search goes on
        interrupted(GAConfig(**SMALL), Path(b), fake_trainer([], fail=lambda n: n == first), gen1 + 4)
        assert sum(r["status"].startswith("error") for r in lines(Path(b) / "results.jsonl")) == 1
        calls = []
        quiet(GAConfig(**SMALL), Path(b), fake_trainer(calls)).run()
        rb = [(r["hash"], r["eval"]) for r in lines(Path(b) / "results.jsonl")]
        assert sorted(rb, key=lambda t: t[1]) == ra  # the one trained again is appended after the reused ones
        assert calls[0] == (ra[first][0], 10)  # the failed one first (drawn again from the saved state)

        try:
            quiet(GAConfig(**SMALL), Path(c), fake_trainer([], fail=lambda n: n >= gen1)).run()
            raise AssertionError("not stopped")
        except RuntimeError as e:
            assert "in a row" in str(e)
        assert sum(r["status"].startswith("error") for r in lines(Path(c) / "results.jsonl")) == MAX_ERRORS
        quiet(GAConfig(**SMALL), Path(c), fake_trainer([])).run()
        assert [(r["hash"], r["eval"]) for r in lines(Path(c) / "results.jsonl")] == ra


def test_read_jsonl():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3', encoding="utf-8")
        assert read_jsonl(p, repair=False) == [{"a": 1}, {"a": 2}] and p.read_text(encoding="utf-8").endswith("3")
        assert read_jsonl(p) == [{"a": 1}, {"a": 2}] and p.read_text(encoding="utf-8").endswith("}\n")
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"a": 4}\n')
        assert read_jsonl(p) == [{"a": 1}, {"a": 2}, {"a": 4}]
        gz = Path(d) / "y.jsonl"  # only a published copy, y.jsonl.gz: readers read it, the writer does not
        gz.with_name("y.jsonl.gz").write_bytes(gzip.compress(b'{"a": 1}\n{"a": 2}\n'))
        assert read_jsonl(gz, repair=False) == [{"a": 1}, {"a": 2}] and read_jsonl(gz) == []
        p.write_text('{"a": 1}\n{"a\n{"a": 3}\n', encoding="utf-8")
        try:
            read_jsonl(p)
            raise AssertionError("a broken line in the middle is not an error")
        except json.JSONDecodeError:
            pass


def test_fewer_elites_than_m():
    """With K < M the mutation children take turns among the elites: still C + M new networks."""
    with tempfile.TemporaryDirectory() as d:
        cfg = GAConfig(**SMALL | {"k": 2, "m": 3})
        quiet(cfg, Path(d), fake_trainer([])).run()
        recs = lines(Path(d) / "results.jsonl")
        for g in range(1, cfg.generations + 1):
            kids = [r for r in recs if r["generation"] == g]
            assert len(kids) == cfg.c + cfg.m and sum(r["origin"] == "mutation" for r in kids) >= 1


def test_pareto_and_entropy():
    recs = [{"hash": h, "macs": m, "val": {"macro_f1": f}} for h, m, f in
            (("a", 1, 0.5), ("b", 2, 0.4), ("c", 3, 0.7), ("d", 3, 0.6), ("e", 5, 0.9), ("f", 6, 0.8))]
    assert [r["hash"] for r in pareto_front(recs, 10)] == ["a", "c", "e"]  # cheaper or better than every other
    assert [r["hash"] for r in pareto_front(recs, 2)] == ["a", "e"]  # spread over the MACs order
    rng = random.Random(0)
    gs = [sample(rng) for _ in range(30)]
    closed = [g | {f"s{r}.active": False for r in SLOTS[1:]} for g in gs]  # only s64 open
    same = [c | {k: closed[0][k] for k in c if k.startswith("s64.") or k.startswith(("stem.", "head."))} for c in closed]
    assert active_entropy(same) == 0.0  # closed slots' genes differ, but nothing that is built


def test_make_rules():
    curves = [[0.1 * i] * 10 for i in range(10)]
    names = lambda rules: [type(r).__name__ + getattr(r, "name", "") for r in rules]  # noqa: E731
    assert names(make_rules(RULES40, [], None, 0.0, 40)) == ["NotLearning", "MinF1floor", "Patience"]
    full = make_rules(RULES40, curves, 70.0, 0.5, 10)  # 10-epoch recipe: later check epochs dropped
    assert names(full) == ["NotLearning", "MinF1floor", "MinF1percentile", "MinF1reach_elite", "Patience"]
    assert set(full[2].thresholds) == {5, 10} and set(full[3].thresholds) == {5, 10}
    off = RULES40 | {"percentile": None, "reach_elite": None}
    assert names(make_rules(off, curves, 70.0, 0.5, 10)) == ["NotLearning", "MinF1floor", "Patience"]
    # every check epoch beyond a 4-epoch recipe: the rules are left out, not replaced by their defaults
    assert names(make_rules(RULES40, [c[:4] for c in curves], 70.0, 0.5, 4)) == ["NotLearning", "Patience"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
