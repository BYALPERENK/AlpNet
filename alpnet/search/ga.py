"""(mu + lambda) genetic algorithm with ageing (README, "Making new networks").

Generation 0: random candidates within budget, trained fully (no stop rules) until N0 complete at the
first phase's epoch count; each is trained with every later phase's epoch count too.
Every later generation:
  1. selection: the K fittest of the population are the elites; the rest die
  2. crossover: C children, each from a random pair of elites (space.crossover, p1 / p2 / pm per gene)
  3. mutation: each of the M fittest elites has one child by mutation alone (space.mutate, qm per gene)
  4. the children are trained with this generation's stop rules, built now from the phase's reference
     curves and the K-th elite's fitness, so no decision depends on the order in which candidates finish
  5. age: elites get one generation older, children start at 0; every elite older than A dies with
     probability DR (the best one too; nothing is lost, the history keeps every evaluation)
  6. population = surviving elites + the children that completed (pruned / failed ones never compete)
A child whose phenotype hash is already in the history (or among this generation's children) or that is
over budget is not trained: it is drawn again (up to `max_tries`, then a random genotype), so every
generation trains C + M new networks.

Epoch phases: the phases follow one another, each trains
the children of its `generations` generations with its own epoch count and stop rules. Scores of
different epoch counts are never compared: a record is one training (hash, epochs), and selection reads
the population's scores at the previous generation's epoch count. At the first generation of a later
phase (a config boundary, whatever scores the population has) the best networks (the elites, chosen with the previous phase's
scores, and the K best of every network trained with that epoch count) are trained again with the new
epoch count (origin "reeval") and the elites are chosen among them.

Stop rule history: a phase's A and B read generation 0's curves at its epoch count (random networks
trained without stop rules, the set the rules were calibrated on), fixed for the whole search. Curves of
completed children would leave out the ones A cut, and the quantiles would rise every generation even
with no better children.

Every evaluation is one line of results.jsonl, written by this process only; the history (cache) is read
from it at start (a last line cut by a crash mid-write is dropped). ga_state.json (generation, population
with ages, RNG state, config) is written after every generation, the previous one kept as
ga_state.prev.json; every write is flushed to disk (the PC may lose power under load). A restart
continues after the last finished generation: the children are drawn again from the saved RNG state, so
they are the same, and those already in results.jsonl are reused instead of retrained (a candidate of the
unfinished generation killed mid-training, or failed with an error, is trained again; one that failed in
a finished generation stays a failure). If the state cannot be read, or results.jsonl lost records of
its generations, the previous state is used (that generation is done again). A restart with another
config is refused, except more generations in the last phase (an extended run), and so is any record
whose cache key (decoder version, dataset, seed, recipe apart from the epochs, memory cap) differs from
the search's. generations.jsonl gets the diagnostics of every generation (lines of an unfinished
generation are dropped at restart).

The training is a function passed in (`evaluate`), so tests can use a fake one. An error that breaks the
GPU state is raised by it (the process stops); MAX_ERRORS errors in a row stop the search too.
"""

import gzip
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..models.decoder import decode
from ..train.stopping import NotLearning, Patience, floor, percentile, reach_elite
from .fitness import cost_penalty
from .space import GENES, SLOT_GENES, SLOTS, crossover, mutate, sample

MAX_ERRORS = 5  # trainings in a row failed with an error (not OOM / NaN): something systemic, stop


def cache_key(recipe: dict, dataset: str, seed: int, decoder_version: int | None, mem_cap_gb: float | None = None) -> dict:
    """What makes two trainings of the same (hash, epochs) the same.
    dataset: loader.dataset_id (name and content hash); mem_cap_gb: the PyTorch memory cap (an OOM under a
    small trial cap is not an OOM under the real one)."""
    return {"decoder_version": decoder_version, "dataset": dataset, "seed": seed, "mem_cap_gb": mem_cap_gb,
            "recipe": {k: v for k, v in recipe.items() if k != "epochs"}}


def write_atomic(path: Path, text: str):
    """Temp file flushed to disk, then renamed over `path`: a crash leaves the old or the new file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, rec: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path, repair: bool = True) -> list[dict]:
    """The records of a JSONL file only one process appends to. A last line cut by a crash mid-write is
    left out, and with `repair` removed from the file (that candidate is trained again; a reader running
    next to the writer passes repair=False); a broken line elsewhere is an error. A reader also reads a
    published copy, <path>.gz (results/), when the file itself is not there."""
    if not path.exists():
        gz = path.with_name(path.name + ".gz")
        if repair or not gz.exists():
            return []
        return [json.loads(line) for line in gzip.decompress(gz.read_bytes()).decode("utf-8").splitlines()]
    text = path.read_text(encoding="utf-8")
    lines, recs = text.splitlines(), []
    for i, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    if repair and (len(recs) < len(lines) or (text and not text.endswith("\n"))):  # rewrite whole lines
        write_atomic(path, "".join(json.dumps(r) + "\n" for r in recs))
    return recs


def is_error(r: dict) -> bool:
    """Failed with an exception (not OOM, NaN or a stop rule): not a property of the network, trained
    again at a restart."""
    return r["status"].startswith("error")


@dataclass
class GAConfig:
    n0: int = 100  # generation 0: completed random candidates
    k: int = 20  # elites
    c: int = 30  # crossover children per generation
    p1: float = 0.45  # per gene: from parent 1 / parent 2, else a random value (pm = 1 - p1 - p2)
    p2: float = 0.45
    m: int = 20  # elites with a mutation child
    qm: float = 0.1  # per-gene change probability of a mutation child
    age_max: int = 5  # A
    death_rate: float = 0.3  # DR
    # [{epochs, generations, rules, calibration}] in order: phase i trains the children of the
    # `generations` generations after the previous phases' with `epochs` epochs and its stop rules
    # (stopping.py, given in full: a rule left out is off); calibration: the calibrate_rules.py report the
    # thresholds come from (missing / null: not calibrated, run_search.py warns)
    phases: list = field(default_factory=list)
    seed: int = 0  # the GA's own randomness
    max_tries: int = 20

    def __post_init__(self):
        assert 0 <= self.p1 and 0 <= self.p2 and self.p1 + self.p2 <= 1, "p1 + p2 must be at most 1"
        assert 1 <= self.k <= self.n0 and self.c >= 0 and self.m >= 0, "1 <= k <= n0, c and m >= 0"
        assert 0 <= self.qm <= 1 and 0 <= self.death_rate <= 1 and self.age_max >= 0, "qm, death_rate, age_max"
        assert self.phases, "phases are required (configs/search.yaml)"
        for p in self.phases:
            assert ({"epochs", "generations", "rules"} <= set(p) <= {"epochs", "generations", "rules", "calibration"}
                    and p["generations"] >= 1 and p["epochs"] >= 1 and isinstance(p["rules"], dict)), f"phase {p}"
        epochs = [p["epochs"] for p in self.phases]
        assert len(set(epochs)) == len(epochs), "every phase has its own epoch count"

    @property
    def generations(self) -> int:
        return sum(p["generations"] for p in self.phases)

    @property
    def epoch_counts(self) -> list[int]:
        """The epoch counts generation 0 is trained with, the first phase's first."""
        return [p["epochs"] for p in self.phases]

    def phase(self, gen: int) -> dict:
        """The phase generation `gen` belongs to (generation 0: the first)."""
        end = 0
        for p in self.phases:
            end += p["generations"]
            if gen <= end:
                return p
        raise ValueError(f"generation {gen} is after the last phase")

    def ranges(self) -> list[tuple[int, int, int]]:
        """(first generation, last generation, epochs) of every phase."""
        out, start = [], 1
        for p in self.phases:
            out.append((start, start + p["generations"] - 1, p["epochs"]))
            start += p["generations"]
        return out


def extends(saved: dict, now: dict) -> bool:
    """`now` is the `saved` config (as JSON) with more generations in its last phase: an extended run.
    The generations already done keep their phase, so the run continues where it stopped."""
    if saved.keys() != now.keys() or any(saved[k] != now[k] for k in saved if k != "phases"):
        return False
    a, b = saved["phases"], now["phases"]
    return (len(a) == len(b) and a[:-1] == b[:-1] and {**a[-1], "generations": 0} == {**b[-1], "generations": 0}
            and b[-1]["generations"] >= a[-1]["generations"])


def make_rules(spec: dict, curves: list[list[float]], elite_fitness: float | None, penalty: float,
               epochs: int) -> list:
    """Stop rules for one child, in order: not learning, D, A, B, patience. A and B need reference curves,
    B an elite bar (none in generation 0)."""
    def within(d):
        return {int(e): v for e, v in d.items() if int(e) <= epochs}

    rules = []  # a rule with no check epoch within the recipe is left out
    if (s := spec.get("not_learning")) is not None:
        rules.append(NotLearning(**s))
    if (s := spec.get("floor")) is not None and (t := within(s)):
        rules.append(floor(t))
    if curves and (s := spec.get("percentile")) is not None and (t := within(s)):
        rules.append(percentile(curves, t, epochs))
    if (curves and elite_fitness is not None and (s := spec.get("reach_elite")) is not None
            and (check := tuple(e for e in s["check"] if e <= epochs))):
        rules.append(reach_elite(curves, elite_fitness, penalty, check, s["q"], epochs))
    if (s := spec.get("patience")) is not None:
        rules.append(Patience(**s))
    return rules


def eligible(r: dict) -> bool:
    """Can compete for selection: trained to the end (a patience stop included)."""
    return r["status"] == "completed" and r.get("fitness") is not None


class Interrupted(Exception):
    """Raised after `stop_after` new evaluations (to test a restart)."""


class Search:
    def __init__(self, cfg: GAConfig, out: Path, evaluate, stop_after: int | None = None, log=print,
                 key: dict | None = None):
        """evaluate(idx, genotype, phenotype, rules, gen_dir, epochs) -> results line (train_random.run_one).
        key: cache_key() of the trainings evaluate does; every record must have it (None: not checked)."""
        self.cfg, self.out, self.evaluate, self.log, self.key = cfg, Path(out), evaluate, log, key
        self.stop_after, self.new_evals, self.errors = stop_after, 0, 0
        self.out.mkdir(parents=True, exist_ok=True)
        self.results, self.state_path = self.out / "results.jsonl", self.out / "ga_state.json"
        self.prev_path, self.gens_path = self.out / "ga_state.prev.json", self.out / "generations.jsonl"
        self.rng = random.Random(cfg.seed)
        self.generation, self.population = None, []  # population: [{"hash", "age"}]
        recs = read_jsonl(self.results)
        if key is not None and (bad := [r for r in recs if r.get("key") != key]):
            raise ValueError(f"{self.out}: {len(bad)} records were trained with another cache key "
                             f"({bad[0].get('key')}), this search uses {key}")
        if (st := self._load_state(recs)) is not None:
            now, saved = json.loads(json.dumps(asdict(cfg))), st.get("config") or {}  # as saved (JSON keys are strings)
            if saved != now and not extends(saved, now):
                diff = sorted(k for k in now.keys() | saved.keys() if now.get(k) != saved.get(k))
                raise ValueError(f"{self.out}: started with another config (differs in {', '.join(diff)}); "
                                 "continue with the config it started with (only the last phase may get more "
                                 "generations)")
            if saved != now:
                log(f"extended: the last phase has {cfg.phases[-1]['generations']} generations "
                    f"(was {saved['phases'][-1]['generations']})")
            self.generation, self.population = st["generation"], st["population"]
            v, internal, gauss = st["rng"]
            self.rng.setstate((v, tuple(internal), gauss))
        last = -1 if self.generation is None else self.generation
        if retry := [r for r in recs if r["generation"] > last and is_error(r)]:
            log(f"{len(retry)} evaluations of generation {last + 1} failed with an error: trained again")
            recs = [r for r in recs if not (r["generation"] > last and is_error(r))]
            write_atomic(self.results, "".join(json.dumps(r) + "\n" for r in recs))
        self.history, self.recs, self.genotypes = [], {}, {}  # recs: (hash, epochs) -> record
        for r in recs:
            if r["generation"] <= last:
                self._remember(r)
        self.pending = {(r["hash"], r["epochs"]): r for r in recs if r["generation"] > last}  # unfinished generation
        if self.pending:
            log(f"resuming generation {last + 1}: {len(self.pending)} evaluations reused")
        gens = read_jsonl(self.gens_path)
        if (keep := list({d["generation"]: d for d in gens if d["generation"] <= last}.values())) != gens:
            write_atomic(self.gens_path, "".join(json.dumps(d) + "\n" for d in keep))

    def _load_state(self, recs: list[dict]) -> dict | None:
        """ga_state.json, else the previous generation's: when it cannot be read after a crash, or
        results.jsonl lost records of its generations (fewer than it counted, or a population member
        without its completed training at that phase's epoch count). An unusable ga_state.json is moved
        to ga_state.bad.json, so the next save does not put it in place of the good previous one.
        None before generation 0 is finished."""
        for path in (self.state_path, self.prev_path):
            if not path.exists():
                continue
            try:
                st = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.log(f"{path.name} cannot be read")
                continue
            gen = st["generation"]
            done = [r for r in recs if r["generation"] <= gen]
            e = self.cfg.phase(gen)["epochs"]
            ok = {r["hash"] for r in done if r["epochs"] == e and eligible(r)}
            if missing := [p["hash"] for p in st["population"] if p["hash"] not in ok]:
                self.log(f"{path.name}: {len(missing)} networks of generation {gen} have no completed {e}-epoch "
                         "record in results.jsonl")
                continue
            if len(done) != st["evaluations"]:
                self.log(f"{path.name}: {st['evaluations']} evaluations up to generation {gen}, results.jsonl "
                         f"has {len(done)}")
                continue
            if path == self.prev_path:
                if self.state_path.exists():
                    os.replace(self.state_path, self.out / "ga_state.bad.json")
                self.log(f"continuing from {path.name}: generation {gen + 1} is done again")
            return st
        if self.state_path.exists() or self.prev_path.exists():
            raise ValueError(f"{self.out}: no usable ga_state.json; results.jsonl does not match it")
        return None

    # ---- scores

    def fitness_at(self, h: str, epochs: int) -> float | None:
        """The fitness of network `h` trained with `epochs` epochs; None if not trained so to the end."""
        r = self.recs.get((h, epochs))
        return r["fitness"] if r is not None and eligible(r) else None

    def reference(self, epochs: int) -> list[list[float]]:
        """The stop rules' history of a phase: generation 0's curves at its epoch count (module docstring)."""
        return [r["curve"] for r in self.history
                if r["generation"] == 0 and r["epochs"] == epochs and r["status"] == "completed"]

    # ---- running

    def run(self):
        if self.generation is None:
            self.generation0()
        while self.generation < self.cfg.generations:
            self.step()

    def generation0(self):
        first = self.cfg.epoch_counts[0]
        taken, recs = set(), []
        while sum(eligible(r) for r in recs if r["epochs"] == first) < self.cfg.n0:
            g = sample(self.rng)
            ph = decode(g)
            if not ph.within_budget() or ph.hash in taken:
                continue
            taken.add(ph.hash)
            for e in self.cfg.epoch_counts:
                recs.append(self._train(0, len(taken), g, ph, [], "random", [], e))
        self.population = [{"hash": r["hash"], "age": 0} for r in recs if r["epochs"] == first and eligible(r)]
        self._finish(0, recs, elites=[], bar=None, killed=[], phase=None)

    def import_generation0(self, srcs):
        """Generation 0 trained by scripts/train_random.py (same recipe and candidate seed): one run per
        phase epoch count, of the same random networks (same --seed). results.jsonl is written in one go
        (temp file + rename), then the state."""
        srcs = [Path(s) for s in srcs]
        assert self.generation is None and not self.history, "search already started"
        if self.pending:  # an import cut short before its state was saved: done again from the start
            assert all(r["generation"] == 0 for r in self.pending.values()), "search already started"
            self.pending = {}
        recs = [r for d in srcs for r in self._import(d)]
        if dup := [k for k, n in Counter((r["hash"], r["epochs"]) for r in recs).items() if n > 1]:
            raise ValueError(f"{len(dup)} networks imported twice with the same epoch count")
        got, need = {r["epochs"] for r in recs}, set(self.cfg.epoch_counts)
        if got != need:
            raise ValueError(f"generation 0 runs have {sorted(got)} epochs, the phases need {sorted(need)}")
        first = self.cfg.epoch_counts[0]
        self.population = [{"hash": r["hash"], "age": 0} for r in recs if r["epochs"] == first and eligible(r)]
        if len(self.population) < self.cfg.n0:
            raise ValueError(f"{len(self.population)} networks completed at {first} epochs, n0 is {self.cfg.n0}")
        for e in self.cfg.epoch_counts[1:]:  # the same networks with every epoch count
            have = {r["hash"] for r in recs if r["epochs"] == e}
            if lack := [p["hash"] for p in self.population if p["hash"] not in have]:
                raise ValueError(f"{len(lack)} of the {first}-epoch networks were not trained with {e} epochs: "
                                 "not the same networks (train_random.py --like)")
        for e in self.cfg.epoch_counts:  # A and B of that phase read these curves
            if (n := sum(r["epochs"] == e and r["status"] == "completed" for r in recs)) < self.cfg.n0 // 2:
                raise ValueError(f"only {n} generation 0 networks completed with {e} epochs (at least n0 / 2 = "
                                 f"{self.cfg.n0 // 2}): too few reference curves for the stop rules")
        write_atomic(self.results, "".join(json.dumps(r) + "\n" for r in self.history))
        self._finish(0, recs, elites=[], bar=None, killed=[], phase=None)

    def _import(self, src: Path) -> list[dict]:
        recs = []
        for row in read_jsonl(src / "results.jsonl"):
            run_dir = src / f"{row['idx']:03d}_{row['hash']}"
            if is_error(row):
                self.log(f"{run_dir.name}: {row['status'][:80]}, not imported")
                continue
            if self.key is not None and row["key"] != self.key:
                raise ValueError(f"{run_dir} was trained with cache key {row['key']}, this search uses {self.key}")
            recs.append(self._remember(self._record(row, 0, "random", [], str(run_dir), row["epochs"])))
        return recs

    def step(self):
        cfg, gen = self.cfg, self.generation + 1
        phase = cfg.phase(gen)
        epochs = phase["epochs"]
        # the population was scored with the previous generation's epoch count (generation 0: the first phase's)
        scored_e = cfg.phase(gen - 1)["epochs"]
        pop = sorted((self._member(p, scored_e) for p in self.population), key=lambda r: -r["fitness"])
        elites = pop[:cfg.k]
        recs = []
        if scored_e != epochs:  # first generation of a later phase: the best networks are trained again
            known = {e["hash"] for e in elites}
            done = [r for r in self.history if r["epochs"] == scored_e and eligible(r)]
            best = sorted(done, key=lambda r: -r["fitness"])[:cfg.k]  # dead elites and earlier networks too
            front = pareto_front(done, cfg.k)  # val F1 vs MACs: the cheap and the large good ones (S / M / L)
            cands = elites + [{"hash": h, "age": 0} for h in dict.fromkeys(r["hash"] for r in best + front)
                              if h not in known]
            for i, e in enumerate(cands):
                if (e["hash"], epochs) not in self.recs:
                    g = self.genotypes[e["hash"]]
                    recs.append(self._train(gen, i + 1, g, decode(g), [], "reeval", [e["hash"]], epochs))
            elites = sorted((self._member(e, epochs) for e in cands if self.fitness_at(e["hash"], epochs) is not None),
                            key=lambda r: -r["fitness"])[:cfg.k]
        bar = elites[-1]["fitness"] if len(elites) == cfg.k else None
        curves = self.reference(epochs)

        taken = set()
        # M mutation children from the M fittest elites; with fewer elites they take turns (C + M new networks)
        jobs = [("crossover", None)] * cfg.c + ([("mutation", elites[i % len(elites)]) for i in range(cfg.m)]
                                                if elites else [("crossover", None)] * cfg.m)
        for origin, parent in jobs:
            if origin == "crossover":
                def make():
                    if len(elites) < 2:
                        return sample(self.rng), "random", []
                    a, b = self.rng.sample(elites, 2)
                    return crossover(a["genotype"], b["genotype"], self.rng, cfg.p1, cfg.p2), "crossover", [a["hash"], b["hash"]]
            else:
                def make(parent=parent):
                    return mutate(parent["genotype"], self.rng, cfg.qm), "mutation", [parent["hash"]]
            g, ph, kind, parents = self._draw(make, taken)
            taken.add(ph.hash)
            rules = make_rules(phase["rules"], curves, bar, cost_penalty(ph.params, ph.macs), epochs)
            recs.append(self._train(gen, len(recs) + 1, g, ph, rules, kind, parents, epochs))

        survivors, killed = [], []
        for e in elites:
            if e["age"] + 1 > cfg.age_max and self.rng.random() < cfg.death_rate:
                killed.append({"hash": e["hash"], "age": e["age"] + 1, "fitness": e["fitness"]})
            else:
                survivors.append({"hash": e["hash"], "age": e["age"] + 1})
        kids = [r for r in recs if r["origin"] != "reeval" and eligible(r)]
        self.population = survivors + [{"hash": r["hash"], "age": 0} for r in kids]
        self._finish(gen, recs, elites, bar, killed, phase)

    def _member(self, p: dict, epochs: int) -> dict:
        """A population entry with its genotype and its fitness at `epochs`."""
        return {"hash": p["hash"], "age": p["age"], "genotype": self.genotypes[p["hash"]],
                "fitness": self.fitness_at(p["hash"], epochs)}

    def _draw(self, make, taken):
        """A new, within-budget child: make() again while the phenotype is known, then a random genotype."""
        for _ in range(self.cfg.max_tries):
            g, kind, parents = make()
            ph = decode(g)
            if ph.within_budget() and ph.hash not in self.genotypes and ph.hash not in taken:
                return g, ph, kind, parents
        while True:
            g = sample(self.rng)
            ph = decode(g)
            if ph.within_budget() and ph.hash not in self.genotypes and ph.hash not in taken:
                return g, ph, "random", []

    def _train(self, gen, idx, g, ph, rules, origin, parents, epochs) -> dict:
        if (ph.hash, epochs) in self.pending:
            rec = self._remember(self.pending.pop((ph.hash, epochs)))
        else:
            gen_dir = self.out / f"gen{gen:02d}" / (f"{epochs}ep" if gen == 0 else "reeval" if origin == "reeval" else "")
            row = self.evaluate(idx, g, ph, rules, gen_dir, epochs)
            rec = self._add(self._record(row, gen, origin, parents, str(gen_dir / f"{idx:03d}_{ph.hash}"), epochs))
            self.new_evals += 1
        f = rec.get("fitness")
        self.log(f"gen {gen:2d} #{idx:3d} [{len(self.history):4d}] {origin:9s} {epochs:2d} ep "
                 f"{rec['status'][:60]:60s} " + (f"fitness {f:5.1f}" if f is not None else ""))
        self.errors = self.errors + 1 if is_error(rec) else 0
        if self.errors >= MAX_ERRORS:
            raise RuntimeError(f"{self.errors} evaluations in a row failed with an error (last: {rec['status']}); "
                               "stopped. A restart trains them again.")
        if self.stop_after is not None and self.new_evals >= self.stop_after:
            raise Interrupted(f"stopped after {self.new_evals} evaluations")
        return rec

    def _record(self, row: dict, gen: int, origin: str, parents: list, run_dir: str, epochs: int) -> dict:
        key = row.get("key", self.key)
        if self.key is not None and key != self.key:
            raise ValueError(f"evaluate() trained with cache key {key}, this search uses {self.key}")
        return row | {"generation": gen, "origin": origin, "parents": parents, "epochs": epochs,
                      "eval": len(self.history) + 1, "run_dir": run_dir, "key": key}

    def _remember(self, rec: dict) -> dict:
        self.history.append(rec)
        self.recs[rec["hash"], rec["epochs"]] = rec
        self.genotypes[rec["hash"]] = rec["genotype"]
        return rec

    def _add(self, rec: dict) -> dict:
        append_jsonl(self.results, rec)
        return self._remember(rec)

    def _finish(self, gen, recs, elites, bar, killed, phase):
        self.generation = gen
        diag = self.diagnostics(gen, recs, elites, bar, killed, phase)
        append_jsonl(self.gens_path, diag)  # a restart before the state is saved drops it again
        state = {"generation": gen, "population": self.population, "rng": list(self.rng.getstate()),
                 "evaluations": len(self.history), "config": asdict(self.cfg)}
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(state))
            f.flush()
            os.fsync(f.fileno())
        if self.state_path.exists():
            os.replace(self.state_path, self.prev_path)
        os.replace(tmp, self.state_path)
        self.log(f"generation {gen} done: {len(recs)} trained, population {len(self.population)}, "
                 f"best {diag['best_fitness']:.2f}, history {len(self.history)}")

    # ---- diagnostics

    def diagnostics(self, gen, recs, elites, bar, killed, phase) -> dict:
        epochs = phase["epochs"] if phase else self.cfg.epoch_counts[0]
        pop = [self._member(p, epochs) for p in self.population]
        fit = [r["fitness"] for r in pop]
        kids = [r for r in recs if r["origin"] != "reeval"]
        d = {"generation": gen, "epochs": epochs, "evaluations": len(self.history), "trained": len(recs),
             "reevaluated": len(recs) - len(kids), "population": len(pop), "elites": len(elites), "elite_bar": bar,
             "best_fitness": max(fit, default=float("nan")), "median_fitness": float(np.median(fit)) if fit else None,
             "elite_mean_fitness": float(np.mean([e["fitness"] for e in elites])) if elites else None,
             "best_hash": max(pop, key=lambda r: r["fitness"])["hash"] if pop else None,
             "origins": dict(Counter(r["origin"] for r in recs)),
             "status": dict(Counter(r["status"].split(" at epoch")[0].split(" (")[0] for r in kids)),
             "epochs_trained": sum(r.get("epochs_done", 0) for r in recs),
             "epochs_full": sum(r["epochs"] for r in recs),
             "train_hours": sum(r.get("total_s", 0) for r in recs) / 3600,
             "ages": dict(Counter(p["age"] for p in self.population)), "killed_by_age": killed,
             "rules": phase and phase["rules"]}
        # inheritance: child fitness vs its parents' (mean and best) at this epoch count, completed children
        for origin in ("crossover", "mutation"):
            pairs = [(r["fitness"], [self.fitness_at(p, epochs) for p in r["parents"]]) for r in kids
                     if r["origin"] == origin and eligible(r)]
            pairs = [(c, p) for c, p in pairs if None not in p]
            if len(pairs) >= 3:
                child = [c for c, _ in pairs]
                d[f"inherit_{origin}"] = {"n": len(pairs),
                                          "rho_mean": _rho(child, [np.mean(p) for _, p in pairs]),
                                          "rho_best": _rho(child, [max(p) for _, p in pairs]),
                                          "child_minus_mean": float(np.median([c - np.mean(p) for c, p in pairs]))}
        # diversity: normalized entropy of the genes that build something, and gene statistics
        gs = [r["genotype"] for r in pop]
        if gs:
            d["gene_entropy_mean"] = active_entropy(gs)
            d["gene_stats"] = gene_stats(gs)
        return d


def active_entropy(genotypes: list[dict]) -> float:
    """Mean normalized entropy over the stem / head genes, the active flags and every slot gene among the
    networks where that slot is active (at least 2 of them). Closed slots' genes are left out: they drift
    without selection and would keep the entropy high."""
    ent = []
    for k, v in GENES.items():
        r = k.split(".")[0]
        if k.startswith(("stem.", "head.")) or k.endswith(".active"):
            vals = [g[k] for g in genotypes]
        else:
            vals = [g[k] for g in genotypes if g[f"{r}.active"]]
        if len(vals) >= 2:
            ent.append(_entropy(Counter(vals), len(v)))
    return float(np.mean(ent))


def pareto_front(recs: list[dict], n: int) -> list[dict]:
    """The records no other beats on both val macro-F1 (higher) and MACs (lower); with more than n, n of
    them spread evenly over the MACs order."""
    front, best = [], -1.0
    for r in sorted(recs, key=lambda r: (r["macs"], -r["val"]["macro_f1"])):
        if r["val"]["macro_f1"] > best:
            front.append(r)
            best = r["val"]["macro_f1"]
    if len(front) > n:
        front = [front[round(i * (len(front) - 1) / (n - 1))] for i in range(n)] if n > 1 else front[-1:]
    return front


def gene_stats(genotypes: list[dict]) -> dict:
    """Value counts over the active slots (slot genes) and over the networks (stem / head / active flags)."""
    out = {f"s{r}.active": sum(g[f"s{r}.active"] for g in genotypes) for r in SLOTS}
    for k in SLOT_GENES:
        if k != "active":
            out[k] = dict(Counter(str(g[f"s{r}.{k}"]) for g in genotypes for r in SLOTS if g[f"s{r}.active"]))
    for k in GENES:
        if k.startswith(("stem.", "head.")):
            out[k] = dict(Counter(str(g[k]) for g in genotypes))
    return out


def _entropy(counts: Counter, n_options: int) -> float:
    n = sum(counts.values())
    h = -sum(c / n * math.log(c / n) for c in counts.values())
    return h / math.log(n_options) if n_options > 1 else 0.0


def _rho(a, b) -> float | None:
    if len(set(a)) < 2 or len(set(b)) < 2:  # undefined for constant input
        return None
    return float(spearmanr(a, b)[0])
