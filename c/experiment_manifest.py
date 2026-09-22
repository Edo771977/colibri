#!/usr/bin/env python3
"""Validate reproducible, one-variable Colibri experiment records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _run(record, name):
    run = _object(record.get(name), name)
    config = _object(run.get("config"), f"{name}.config")
    samples = _object(run.get("samples"), f"{name}.samples")
    speeds = samples.get("tok_s")
    if (not isinstance(speeds, list) or len(speeds) < 3 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v <= 0 for v in speeds)):
        raise ValueError(f"{name}.samples.tok_s must contain at least 3 positive finite values")
    median = run.get("median_tok_s")
    if (isinstance(median, bool) or not isinstance(median, (int, float)) or
            not math.isclose(float(median), statistics.median(speeds), rel_tol=1e-6)):
        raise ValueError(f"{name}.median_tok_s must equal the sample median")
    evidence = _object(run.get("evidence"), f"{name}.evidence")
    _text(evidence.get("uri"), f"{name}.evidence.uri")
    digest = evidence.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name}.evidence.sha256 must be 64 hex characters")
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise ValueError(f"{name}.evidence.sha256 must be hexadecimal") from error
    quality = _object(run.get("quality"), f"{name}.quality")
    _text(quality.get("method"), f"{name}.quality.method")
    if quality.get("passed") is not True:
        raise ValueError(f"{name}.quality.passed must be true")
    return config


def validate(record):
    """Return a normalized summary or raise ValueError with an exact field."""
    if record.get("version") != 1:
        raise ValueError("version must be 1")
    for field in ("hypothesis", "commit", "model", "command", "prompt_hash"):
        _text(record.get(field), field)
    commit = record["commit"]
    if len(commit) != 40:
        raise ValueError("commit must be a full 40-character git SHA")
    try:
        bytes.fromhex(commit)
    except ValueError as error:
        raise ValueError("commit must be hexadecimal") from error
    hardware = _object(record.get("hardware"), "hardware")
    for field in ("cpu", "ram", "storage", "os"):
        _text(hardware.get(field), f"hardware.{field}")
    warmup = record.get("warmup_runs")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_runs must be a non-negative integer")

    baseline = _run(record, "baseline")
    trial = _run(record, "trial")
    keys = sorted(set(baseline) | set(trial))
    actual = [key for key in keys if baseline.get(key) != trial.get(key)]
    declared = record.get("changed_variables")
    if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], str):
        raise ValueError("changed_variables must name exactly one variable")
    if actual != declared:
        raise ValueError(f"changed_variables {declared!r} does not match config diff {actual!r}")
    outcome = record.get("outcome")
    if outcome not in ("improvement", "regression", "no-change"):
        raise ValueError("outcome must be improvement, regression, or no-change")
    return {
        "variable": actual[0],
        "baseline_tok_s": record["baseline"]["median_tok_s"],
        "trial_tok_s": record["trial"]["median_tok_s"],
        "outcome": outcome,
    }


def _evidence_path(uri, manifest_path):
    """The repo file an evidence uri names, or None if it names no file.

    The uri is repo-relative and may carry a section after a comma
    ("docs/experiments/x-raw.txt, section 'SESSION 2', arm r0"); only the part
    before the first comma is a path.

    Returning None means "this uri is not a path in this tree": it is empty,
    or it carries a scheme (`manifest.example.json` points at an artifact
    URL). EVERYTHING else is treated as a path, and a path that is not there
    raises. That is the case the digest gate exists for: the raw records of
    this repository get renamed and corrected, and a silent skip on a stale
    uri would leave a manifest looking like a chain of custody while nothing
    verified it. (Until 22 September 2026 this function returned None for
    both, and the caller could not tell them apart. It also dropped any uri
    containing a space -- a heuristic for the example manifest's literal
    "artifact URL", which would equally have dropped a real path with a space
    in it. The example now carries a real scheme instead, so no heuristic is
    needed.)
    """
    head = uri.split(",")[0].strip()
    if not head or "://" in head:
        return None
    for parent in Path(manifest_path).resolve().parents:
        candidate = parent / head
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"evidence.uri names a path that is not in the tree: {head!r} "
        f"(searched upward from {Path(manifest_path).resolve().parent})")


def verify_evidence(record, manifest_path):
    """Check that each arm's sha256 is the digest of the file its uri names.

    Until 22 September 2026 the validator checked only that the digest was 64
    hex characters. Nothing checked that it was the digest of anything, and
    four of the six real manifests in docs/experiments/ had drifted: three
    DSv4.1 records declared DIFFERENT digests for baseline and trial while
    both pointed at one file -- a file has one digest, so those values never
    certified it -- and qwen36-i8-rows.json still carried the digest the raw
    record had before three corrections, one of which fixed the hardware and
    one the run duration. A manifest that certifies a version of a record that
    said the wrong GPU is worse than no manifest, because it looks like a
    chain of custody.

    An evidence uri that does not name a path at all (an artifact URL) is
    skipped: this verifies what it can reach, and says nothing about the rest.
    A uri that names a path which is not in the tree is an error, not a skip.
    """
    for name in ("baseline", "trial"):
        evidence = record.get(name, {}).get("evidence", {})
        path = _evidence_path(evidence.get("uri", ""), manifest_path)
        if path is None:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        declared = (evidence.get("sha256") or "").lower()
        if digest == declared:
            continue
        # A digest is over bytes, so a checkout that rewrote the line endings
        # fails here with nothing to distinguish it from a tampered record.
        # .gitattributes pins these files with `-text` for exactly that
        # reason; if the pin is ever lost, say so instead of accusing the
        # record. The comparison stays byte-exact -- this only names the
        # cause.
        if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == declared:
            raise ValueError(
                f"{name}.evidence.sha256 does not match {path.name} BECAUSE "
                f"THIS CHECKOUT REWROTE THE LINE ENDINGS: the file matches "
                f"the declared digest once CRLF is folded back to LF. The "
                f"record is intact; the working copy is not. Check that "
                f".gitattributes still carries `docs/experiments/*.txt "
                f"-text`, then re-checkout the file.")
        raise ValueError(
            f"{name}.evidence.sha256 does not match {path.name}: "
            f"declared {evidence.get('sha256')}, file is {digest}")


def validate_path(path):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    summary = validate(record)
    verify_evidence(record, path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args(argv)
    failed = False
    for name in args.manifests:
        try:
            summary = validate_path(name)
            print(f"{name}: ok ({summary['variable']}, {summary['outcome']})")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed = True
            print(f"{name}: {error}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
