import copy
import json
import hashlib
import pathlib
import re
import subprocess
import tempfile
import unittest

from experiment_manifest import validate, validate_path


def run(config, speeds):
    return {
        "config": config,
        "samples": {"tok_s": speeds},
        "median_tok_s": sorted(speeds)[1],
        "quality": {"method": "token-exact oracle", "passed": True},
        "evidence": {"uri": "https://example.invalid/raw.log",
                     "sha256": "ab" * 32},
    }


def _pattern_to_regex(pattern, anchored):
    """git's pathspec globbing: `*` stops at `/`, `**` does not."""
    out, i, n = [], 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if pattern[i:i + 1] == "/":
                    i += 1
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.index("]", i) if "]" in pattern[i:] else n - 1
            out.append(pattern[i:j + 1])
            i = j + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    body = "".join(out)
    return re.compile(("" if anchored else "(?:.*/)?") + body + r"\Z")


def _text_attr(root, rel):
    """What `git check-attr text` would report for a repo-relative path.

    Files deeper in the tree win, and inside one file the last matching line
    wins. Implemented here rather than shelled out so the check still means
    something when the tests run from an export with no .git.
    """
    parts = pathlib.PurePosixPath(rel).parts
    verdict = "unspecified"
    for depth in range(len(parts)):                 # root first, deepest last
        directory = root.joinpath(*parts[:depth])
        attrs = directory / ".gitattributes"
        if not attrs.is_file():
            continue
        below = "/".join(parts[depth:])
        for line in attrs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith('"'):
                closing = line.index('"', 1)
                pattern, rest = line[1:closing], line[closing + 1:]
            else:
                pattern, _, rest = line.partition(" ")
            attributes = rest.split()
            if not attributes:
                continue
            anchored = "/" in pattern.rstrip("/")
            if pattern.startswith("/"):
                pattern = pattern[1:]
                anchored = True
            if not _pattern_to_regex(pattern, anchored).match(below):
                continue
            for a in attributes:
                if a == "text" or a.startswith("text=") or a.startswith("eol="):
                    verdict = "set"
                elif a in ("-text", "binary"):
                    verdict = "unset"
                elif a == "!text":
                    verdict = "unspecified"
    return verdict


def manifest():
    return {
        "version": 1,
        "hypothesis": "two loader lanes improve cold decode",
        "commit": "12" * 20,
        "model": "GLM-5.2 int4",
        "command": "NGEN=16 PROF=1 ./coli run ...",
        "prompt_hash": "sha256:example",
        "hardware": {"cpu": "example", "ram": "128 GB",
                     "storage": "NVMe ext4", "os": "Linux"},
        "warmup_runs": 1,
        "changed_variables": ["PIPE_WORKERS"],
        "baseline": run({"PIPE_WORKERS": "1", "DIRECT": "1"}, [1.0, 1.1, 1.2]),
        "trial": run({"PIPE_WORKERS": "2", "DIRECT": "1"}, [1.2, 1.3, 1.4]),
        "outcome": "improvement",
    }


class ExperimentManifestTest(unittest.TestCase):
    def test_accepts_reproducible_one_variable_record(self):
        result = validate(manifest())
        self.assertEqual(result["variable"], "PIPE_WORKERS")

    def test_rejects_hidden_second_variable(self):
        record = manifest()
        record["trial"]["config"]["DIRECT"] = "0"
        with self.assertRaisesRegex(ValueError, "config diff"):
            validate(record)

    def test_rejects_claimed_median_not_backed_by_samples(self):
        record = manifest()
        record["trial"]["median_tok_s"] = 9.9
        with self.assertRaisesRegex(ValueError, "sample median"):
            validate(record)

    def test_rejects_missing_quality_gate(self):
        record = manifest()
        record["trial"]["quality"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "passed must be true"):
            validate(record)

    def test_rejects_unhashed_raw_evidence(self):
        record = copy.deepcopy(manifest())
        record["baseline"]["evidence"]["sha256"] = "unknown"
        with self.assertRaisesRegex(ValueError, "64 hex"):
            validate(record)


class ShippedManifests(unittest.TestCase):
    """Every manifest in docs/experiments/ must actually validate.

    Nothing ran the validator on them until 22 September 2026: the test above
    exercises a synthetic record and CONTRIBUTING.md documents the CLI as a
    manual step. The result was four of six with a sha256 that was not the
    digest of the file it named -- three DSv4.1 records declaring DIFFERENT
    digests for baseline and trial while both named ONE file -- and one with a
    median that did not match its own samples. All five were found by running
    this, which is the argument for it being a test rather than a habit.
    """

    def manifests(self):
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            d = parent / "docs" / "experiments"
            if d.is_dir():
                return sorted(d.glob("*.json"))
        self.skipTest("docs/experiments not found from the test directory")

    def test_every_shipped_manifest_validates(self):
        found = self.manifests()
        self.assertTrue(found, "docs/experiments/ has no manifests to check")
        for path in found:
            with self.subTest(manifest=path.name):
                validate_path(path)


class EvidenceDigest(unittest.TestCase):
    """The digest has to be the digest OF something, not 64 hex characters.

    The old check accepted any well-formed hex string, so a raw record could be
    corrected under a manifest that went on certifying the version before the
    correction -- in one case a version that named the wrong GPU.
    """

    def _written(self, tmp, body):
        """Write the raw record where a repo-relative uri would find it.

        `_evidence_path` walks up from the manifest and joins the uri, so the
        fixture has to reproduce that shape: manifest at the root of `tmp`,
        record under `tmp/docs/experiments/`. A bare filename next to the
        manifest would also resolve, but no manifest in this repository is
        written that way.
        """
        d = pathlib.Path(tmp) / "docs" / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        raw = d / "rec-raw.txt"
        raw.write_bytes(body)
        return raw, hashlib.sha256(body).hexdigest()

    def test_matching_digest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, digest = self._written(tmp, b"arm A 10 tok/s\narm B 12 tok/s\n")
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {"uri": "docs/experiments/" + raw.name, "sha256": digest}
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            validate_path(path)

    def test_stale_digest_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, digest = self._written(tmp, b"before the correction\n")
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {"uri": "docs/experiments/" + raw.name, "sha256": digest}
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            validate_path(path)                      # green before the edit
            raw.write_bytes(b"after the correction\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_path(path)                  # and red after it

    def test_uri_with_a_scheme_is_skipped_not_failed(self):
        """manifest.example.json points at an artifact URL; not a failure."""
        with tempfile.TemporaryDirectory() as tmp:
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {
                    "uri": "https://example.invalid/run-artifact.txt",
                    "sha256": "ab" * 32,
                }
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            validate_path(path)

    def test_uri_naming_a_missing_path_is_an_error_not_a_skip(self):
        """The branch the gate exists for: a record renamed, the uri stale.

        Until 22 September 2026 this was a silent skip -- `make check` stayed
        green and nothing verified the digest, which is the exact condition a
        chain of custody is supposed to exclude. No test covered it: the one
        that claimed to ("artifact URL") exited on a different branch.
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw, digest = self._written(tmp, b"arm A 10 tok/s\n")
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {
                    "uri": "docs/experiments/" + raw.name + "x",
                    "sha256": digest,
                }
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not in the tree"):
                validate_path(path)

    def test_crlf_checkout_is_named_as_the_cause(self):
        """The Windows failure mode, reported as itself and not as tampering.

        `make check` went red on the Windows UCRT64 runner for all six real
        manifests the moment digests started being verified: git had rewritten
        the records to CRLF, so the bytes -- and the digest -- changed. The
        bytes are pinned in .gitattributes now; this covers the case where
        that pin is lost, because a bare "does not match" would read as a
        corrupted record.
        """
        with tempfile.TemporaryDirectory() as tmp:
            body = b"arm A 10 tok/s\narm B 12 tok/s\n"
            raw, digest = self._written(tmp, body)
            raw.write_bytes(body.replace(b"\n", b"\r\n"))   # what Windows checks out
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {
                    "uri": "docs/experiments/" + raw.name,
                    "sha256": digest,
                }
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REWROTE THE LINE ENDINGS"):
                validate_path(path)

    def test_records_are_pinned_against_eol_rewriting(self):
        """The pin itself: without it the digests are not portable.

        Checked by EFFECT. Two earlier versions of this test did not check
        the effect at all. The first looked for the literal line
        `docs/experiments/*.txt -text` and passed with that rule COMMENTED
        OUT. The second resolved it with `fnmatch`, whose `*` crosses `/`,
        which knows nothing about .gitattributes files in subdirectories --
        git gives those precedence -- and which ignores `!attr`; a review
        showed three ways to unpin the records with that version still green,
        including a `docs/experiments/.gitattributes` saying
        `*.txt text eol=crlf`, the exact regression this test exists for.

        This one implements git's own resolution: every .gitattributes from
        the repository root down to the file's own directory, deeper files
        winning, last matching line in each winning, `*` not crossing `/`, a
        leading `/` anchoring, and `!text` returning the attribute to
        unspecified. It asks git for a second opinion when git is available,
        and fails if the two disagree -- a mismatch means this resolver has
        drifted from the thing it models.
        """
        root = pathlib.Path(__file__).resolve().parent.parent.parent
        named = set()
        for path in sorted((root / "docs" / "experiments").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            for arm in ("baseline", "trial"):
                uri = record.get(arm, {}).get("evidence", {}).get("uri", "")
                head = uri.split(",")[0].strip()
                if head and "://" not in head:
                    named.add(head)
        self.assertTrue(named, "no manifest names a file; this test is vacuous")

        unpinned = sorted(r for r in named if _text_attr(root, r) != "unset")
        self.assertEqual(
            unpinned, [],
            "these records are not pinned against end-of-line rewriting, so "
            "their digests are not portable and a Windows checkout will fail "
            "every one of them: " + ", ".join(unpinned))

        # Second opinion, when there is a git to ask.
        try:
            out = subprocess.run(
                ["git", "check-attr", "text", "--"] + sorted(named),
                cwd=root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):       # pragma: no cover
            return
        if out.returncode != 0:
            return                                          # not a checkout
        for line in out.stdout.splitlines():
            rel, _, verdict = line.rpartition(": ")
            rel = rel.rsplit(": ", 1)[0]
            self.assertEqual(
                verdict, "unset",
                f"git says `text` is {verdict!r} for {rel}, so this checkout "
                "rewrites it and its digest cannot match")
            self.assertEqual(
                _text_attr(root, rel), verdict,
                f"this test's attribute resolver disagrees with git on {rel}: "
                f"it says {_text_attr(root, rel)!r}, git says {verdict!r}")

    def test_digest_comparison_ignores_hex_case(self):
        """An uppercase digest is the same digest, not a mismatch."""
        with tempfile.TemporaryDirectory() as tmp:
            raw, digest = self._written(tmp, b"arm A 10 tok/s\n")
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {
                    "uri": "docs/experiments/" + raw.name,
                    "sha256": digest.upper(),
                }
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            validate_path(path)


if __name__ == "__main__":
    unittest.main()
