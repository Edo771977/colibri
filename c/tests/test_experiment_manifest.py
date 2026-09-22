import copy
import json
import os
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
    """git's .gitattributes globbing.

    `*` and `?` stop at `/`. `**` spans whole path components: `a/**/b`
    matches `a/b` and `a/x/y/b` but NOT `a/xb`, which is where a version of
    this that translated `**` to `.*` silently unpinned every record.
    `[!abc]` is a NEGATED class -- git's spelling of `[^abc]`, and reading it
    as a positive one inverted the match. A class never matches `/`, whatever
    it contains.

    APPROXIMATION, in the safe direction: git strips a pattern's literal
    prefix before matching, so in `docs/x**/a.txt` the run of stars ends up
    at position 0 of what wildmatch sees and IS treated as a whole
    component. This resolver applies the whole-component test to the
    original pattern, so it reads that as an ordinary `*` and reports
    unspecified where git says unset. That makes a pinned record look
    unpinned -- the pin test goes red rather than quietly passing -- so it is
    left unmodelled rather than half-modelled. Backslash escapes inside a
    pattern are unmodelled for the same reason and in the same direction.
    """
    out, i, n = [], 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            # `**` is special ONLY where it is a whole path component: at the
            # start or after a `/`, AND at the end or before a `/`. Anywhere
            # else git reads consecutive asterisks as an ordinary `*`, which
            # does not cross `/`. Treating `docs/**.txt` as `.*` made this
            # resolver call a record pinned that git leaves UNSPECIFIED -- a
            # false green, in the no-git case where nothing contradicts it.
            # Found by review, 22 September 2026.
            # git's wildmatch consumes the whole RUN of asterisks and then
            # asks whether the run is a full path component, so `***` and
            # `****` mean `**`. A version of this that recognised only a run
            # of exactly two left `***/*.txt` matching nothing across `/`,
            # which reports a pinned record as unpinned. Found by review.
            run = i
            while run < n and pattern[run] == "*":
                run += 1
            stars = run - i
            whole = (stars >= 2
                     and (i == 0 or pattern[i - 1] == "/")
                     and pattern[run:run + 1] in ("", "/"))
            if whole:
                i = run
                if pattern[i:i + 1] == "/":
                    out.append("(?:[^/]+/)*")       # zero or more components
                    i += 1
                else:
                    out.append(".*")                # trailing ** : everything
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 2)            # `]` right after `[`/`[!`
            if j < 0:
                out.append(re.escape(c))
                i += 1
                continue
            body = pattern[i + 1:j]
            if body.startswith("!"):
                body = "^" + body[1:]
            # Una classe non attraversa MAI `/` in git (wildmatch sotto
            # WM_PATHNAME), nemmeno se il `/` e' dentro la classe o dentro
            # un intervallo. Copiandola nella regex tale e quale,
            # `a[!b]c` matchava `a/c`, che git lascia unspecified.
            out.append("(?:(?![/])[" + body + "])")
            i = j + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    body = "".join(out)
    return re.compile(("" if anchored else "(?:.*/)?") + body + r"\Z")


_ATTR_CHARS = re.compile(r"[-A-Za-z0-9_.]+\Z")


def _valid_attr(token):
    """Does git accept this token as an attribute, or throw the LINE away?

    Measured against `git check-attr` rather than read off the docs, because
    the first version of this function was written from the docs and was
    wrong in both directions: it rejected `_x` and `.x`, which git honours,
    and it accepted `--text` and `-!text`, which git rejects -- the second
    being the false green the check was added to stop, since `--text` is the
    ordinary typo for `-text`.

    git strips AT MOST ONE leading `-` or `!`, then requires the rest to be
    non-empty, to not begin with `-`, and to be made of letters, digits,
    hyphens, underscores and dots. A `=value` suffix is not part of the name.
    Verified on: _x .x _ ... __init__ a-b a.b_c 1abc x= -x !x x- x. A
    (honoured) and x!y --x -!x (thrown away).
    """
    name = token.partition("=")[0]
    if name[:1] in ("-", "!"):
        name = name[1:]
    return bool(name) and not name.startswith("-") and bool(_ATTR_CHARS.match(name))


def _attrs_for(root, rel, wanted):
    """Resolve `wanted` attributes for a repo-relative path, as git does.

    Every .gitattributes from the repository root down to the file's own
    directory, deeper files winning, last matching line in each winning.
    Implemented here rather than shelled out so the check still means
    something when the tests run from an export with no .git -- which is
    exactly where a resolver that has drifted from git does the most damage,
    because there is nothing to contradict it.
    """
    parts = pathlib.PurePosixPath(rel).parts
    verdict = {name: "unspecified" for name in wanted}
    binary_ridefinita = False
    for depth in range(len(parts)):                 # root first, deepest last
        attrs = root.joinpath(*parts[:depth]) / ".gitattributes"
        if not attrs.is_file():
            continue
        below = "/".join(parts[depth:])
        # split("\n") e non splitlines(): quest'ultimo spezza su otto
        # caratteri che git NON considera fine riga (\x0b \x0c \x1c \x1d
        # \x1e \x85 U+2028 U+2029), e su uno di quelli il resolver leggeva
        # due righe dove git ne legge una sola e la butta via -- un pin
        # dichiarato che git non da'. utf-8-sig perche' un BOM, che su
        # Windows e' il modo ordinario in cui arriva, non e' spazio e
        # sopravviveva a strip() rendendo il primo pattern irriconoscibile.
        for line in attrs.read_text(encoding="utf-8-sig").split("\n"):
            # strip(" \t\r") e non strip(): quest'ultimo toglie anche NBSP
            # e gli altri spazi Unicode, che per git sono parte del pattern.
            line = line.strip(" \t\r")
            if not line or line.startswith("#"):
                continue
            closing = line.find('"', 1) if line.startswith('"') else -1
            if closing >= 0:
                pattern, rest = line[1:closing], line[closing + 1:]
            else:
                # Any whitespace separates the pattern from its attributes.
                # A TAB there is ordinary .gitattributes spelling, and
                # partitioning on a literal space missed it -- which failed
                # the check on a correctly pinned tree.
                # split(None) separa su QUALUNQUE spazio Unicode; git
                # separa solo su spazio e tab. Con un NBSP fra pattern e
                # attributi git non vede una riga valida, il resolver si'.
                fields = re.split(r"[ \t]+", line, 1)
                pattern = fields[0]
                rest = fields[1] if len(fields) > 1 else ""
            if pattern == "[attr]binary":
                # git permette di RIDEFINIRE la macro predefinita. Con
                # `[attr]binary text` il pin si inverte e git accende la
                # conversione, mentre il trattamento fisso qui sotto
                # continuerebbe a dichiarare `unset`: un falso verde. Non
                # sapendo espandere le macro utente, da qui in poi `binary`
                # non viene piu' trattato come la macro predefinita, e il
                # risultato cade su unspecified -- la direzione sicura.
                binary_ridefinita = True
                continue
            if pattern.startswith("[attr]"):
                # A macro DEFINITION, not a pattern. git applies it wherever
                # the macro name is later used; this resolver models only the
                # built-in `binary` and cannot expand a user macro, so it
                # skips the definition. A file pinned ONLY through a user
                # macro therefore reads unspecified here and the pin test
                # FAILS -- the safe direction, and the reason this is a
                # documented non-model rather than a silent wrong parse. An
                # earlier version read `[attr]myrec` as a pattern containing
                # a character class.
                continue
            # Anche QUI solo spazio e tab: rest.split() spezzava su NBSP e
            # compagnia, cosi' `-text\u00a0x` diventava due token validi
            # dove git ne vede uno solo, invalido, e butta via la riga.
            # Correggere il separatore fra pattern e attributi senza
            # correggere questo lasciava in piedi 13 falsi verdi su 900.
            attributes = [a for a in re.split(r"[ \t]+", rest) if a]
            if not attributes:
                continue
            # git rejects a line carrying a token that is not a valid
            # attribute name -- letters, digits, hyphens, underscores and
            # dots, starting with a letter or digit. This matters for the
            # ordinary mistake `*.txt -text # note`: there is NO inline
            # comment syntax in .gitattributes, so git throws the whole line
            # away and the records are left unpinned. Accepting it here made
            # the resolver report a pin git does not give. Found by review,
            # 22 September 2026.
            if not all(_valid_attr(a) for a in attributes):
                continue
            anchored = "/" in pattern.rstrip("/")
            if pattern.startswith("/"):
                pattern = pattern[1:]
                anchored = True
            if not _pattern_to_regex(pattern, anchored).match(below):
                continue
            for a in attributes:
                if a == "binary" and not binary_ridefinita:   # macro: -diff -merge -text
                    if "text" in verdict:
                        verdict["text"] = "unset"
                    continue
                name, eq, value = a.partition("=")
                unset = name.startswith("-")
                unspecify = name.startswith("!")
                name = name.lstrip("-!")
                if name not in verdict:
                    continue
                if unspecify:
                    verdict[name] = "unspecified"
                elif unset:
                    verdict[name] = "unset"
                elif eq:
                    verdict[name] = value           # git prints the value
                else:
                    verdict[name] = "set"
    return verdict


_GIT_ISOLATED = None


def _git_env():
    """An environment where git reads ONLY the repository's .gitattributes.

    Without this, `git check-attr` also consults core.attributesFile and the
    system gitattributes, which _attrs_for does not model -- so a developer
    whose global config carries the widely recommended `* text=auto` saw this
    test fail on a correctly pinned tree with a correct resolver. That is the
    false red the resolver exists to avoid, reintroduced by the check meant
    to verify it. Found by review, 22 September 2026.
    """
    global _GIT_ISOLATED
    if _GIT_ISOLATED is None:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_SYSTEM"] = os.devnull
        env["GIT_ATTR_NOSYSTEM"] = "1"
        _GIT_ISOLATED = env
    return _GIT_ISOLATED


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
            with self.assertRaises(ValueError) as caught:
                validate_path(path)                  # and red after it
            # NOT just "does not match": the CRLF diagnostic below ALSO says
            # "does not match", so a regex that loose was satisfied by either
            # branch. With it, `if <crlf-check>:` could be replaced by
            # `if True:` and this test stayed green -- which is the failure
            # where a genuinely edited record is reported as an intact record
            # in a bad working copy, and the reader is told to re-checkout
            # instead of to investigate. Found by review, 22 September 2026.
            self.assertRegex(str(caught.exception), r"declared \w+, file is \w+")
            self.assertNotIn("REWROTE THE LINE ENDINGS", str(caught.exception))

    def test_uri_that_is_all_section_and_no_file_is_an_error(self):
        """", section 'SESSION 2'" names no file. That is a half-finished
        rename, not a uri to skip.

        `validate()` only checks the uri is a non-empty string, so this gets
        as far as `_evidence_path`, which used to return None for it -- a
        SILENT SKIP, the hole the digest gate exists to close, entered from
        the other side. The raise that closes it shipped without a test, so
        replacing it with `return None` left the whole suite green. Found by
        review, 22 September 2026.
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw, digest = self._written(tmp, b"arm A 10 tok/s\n")
            record = copy.deepcopy(manifest())
            for arm in ("baseline", "trial"):
                record[arm]["evidence"] = {
                    "uri": ", section 'SESSION 2', arm r0", "sha256": digest}
            path = pathlib.Path(tmp) / "m.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "names no file, only a section"):
                validate_path(path)

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
        unspecified. Where git is available it also asks git what IT makes of
        the records, and fails if git says they are not `unset`.

        It does NOT compare the two answers here: on this tree both are
        pinned to "unset" by the assertion above, so such a comparison could
        never fail, and an earlier version of this test made it anyway. The
        real comparison is test_resolver_agrees_with_git_on_constructed_cases,
        which builds trees where the two CAN differ.
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

        # Both attributes, because either one alone can rewrite the bytes:
        # `text` turns conversion on, and `eol=` forces a specific ending
        # even where git reports `text` as unspecified.
        unpinned = []
        for rel in sorted(named):
            attrs = _attrs_for(root, rel, ("text", "eol"))
            if attrs["text"] != "unset" or attrs["eol"] != "unspecified":
                unpinned.append(f"{rel} (text: {attrs['text']}, "
                                f"eol: {attrs['eol']})")
        self.assertEqual(
            unpinned, [],
            "these records are not pinned against end-of-line rewriting, so "
            "their digests are not portable and a Windows checkout will fail "
            "every one of them: " + ", ".join(unpinned))

        # Second opinion, when there is a git to ask.
        try:
            out = subprocess.run(
                ["git", "check-attr", "text", "--"] + sorted(named),
                cwd=root, capture_output=True, text=True, timeout=30,
                env=_git_env())
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
        # There used to be a second assertion here, comparing _attrs_for's
        # answer on these same paths with git's. It could not fail: the loop
        # above has already asserted _attrs_for says "unset" for every one of
        # them, and the line above asserts git says "unset" too, so both
        # operands were pinned to the same constant. Four separate drifts of
        # the resolver -- `*` crossing `/`, `**` read as `.*`, `[!abc]` read
        # as a positive class, and only the root .gitattributes being read --
        # all left it green. The real comparison lives in
        # test_resolver_agrees_with_git_on_constructed_cases below, which
        # builds trees where the two CAN differ. Found by review, 22
        # September 2026.

    def test_resolver_agrees_with_git_on_constructed_cases(self):
        """_attrs_for against real `git check-attr`, on trees built to differ.

        The pin test resolves .gitattributes by hand rather than shelling
        out, so that it still means something when the suite runs from an
        export with no .git -- which is exactly where a drifted resolver does
        the most damage, because nothing can contradict it. That only helps
        if the reimplementation is checked against git SOMEWHERE, on inputs
        where the two can actually disagree. The repository's own tree is not
        such an input: one rule, one shape, everything `unset`.

        Every case below is one a previous version of this resolver got
        wrong, or one a review constructed to break it.
        """
        cases = [
            # (lines for <root>/.gitattributes, lines for docs/experiments/,
            #  path to ask about)
            (["docs/experiments/*.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text"], [], "docs/experiments/s/a.txt"),
            (["*.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/**/*.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/**/*.txt -text"], [], "docs/xexperiments.txt"),
            (["docs/**.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text # note"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt\t-text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text", "docs/experiments/*.txt text"],
             [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text"], ["*.txt text eol=crlf"],
             "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text"], ["*.txt !text"],
             "docs/experiments/a.txt"),
            (["docs/experiments/*.txt binary"], [], "docs/experiments/a.txt"),
            (["docs/experiments/[!x]*.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/[!x]*.txt -text"], [], "docs/experiments/x.txt"),
            (["/docs/experiments/*.txt -text"], [], "docs/experiments/a.txt"),
            (["# docs/experiments/*.txt -text"], [], "docs/experiments/a.txt"),
            ([], [], "docs/experiments/a.txt"),
            # Casi aggiunti dalla round 7, ciascuno perche' una mutazione del
            # resolver ci dormiva dentro o perche' la regola era sbagliata.
            (["***/*.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/*** -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt _x -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt .x -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt --text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -!text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt x!y -text"], [], "docs/experiments/a.txt"),
            # NIENTE macro utente qui: questo resolver non le espande, lo
            # dichiara, e fallisce nella direzione sicura (un file pinnato
            # SOLO tramite macro risulta non pinnato e il test del pin
            # diventa rosso). Metterlo nella lista dell'accordo con git
            # pretenderebbe un'implementazione che non c'e'. Che la riga di
            # DEFINIZIONE non venga letta come pattern e' verificato da
            # test_macro_definition_is_not_read_as_a_pattern.
            (["[attr]binary -text"], [], "docs/experiments/a.txt"),
            (["experiments/*.txt -text"], [], "docs/experiments/a.txt"),
            (['"docs/experiments/a b.txt" -text'], [], "docs/experiments/a b.txt"),
            (["docs/experiments/*.txt text=auto"], [], "docs/experiments/a.txt"),
            (["docs/experiments/?.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/experiments/?.txt -text"], [], "docs/experiments/ab.txt"),
            # Casi aggiunti dalla round 8. Ognuno uccide una mutazione che
            # la lista precedente attraversava senza svegliarsi, o copre un
            # falso verde che il fuzz da 400 casi non raggiungeva.
            (["docs?experiments/a.txt -text"], [], "docs/experiments/a.txt"),
            (["docs/* -text"], [], "docs/experiments/a.txt"),
            (["x** -text"], [], "x/y/z.txt"),
            (["a[!b]c -text"], [], "a/c"),
            (["a[/]c -text"], [], "a/c"),
            (['"docs/experiments/a b.txt -text'], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt\u00a0-text"], [], "docs/experiments/a.txt"),
            # NBSP FRA GLI ATTRIBUTI, non fra pattern e attributi: git vede
            # un token solo e invalido e butta la riga. Senza questo caso,
            # riportare rest.split() alla versione Unicode passava inosservato.
            (["docs/experiments/*.txt -text\u00a0x"], [], "docs/experiments/a.txt"),
            (["docs/experiments/*.txt -text\u2028*.txt -text"], [],
             "docs/experiments/a.txt"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for n, (top, deep, rel) in enumerate(cases):
                root = pathlib.Path(tmp) / f"case{n}"
                (root / "docs" / "experiments" / "s").mkdir(parents=True)
                # I casi nominano percorsi arbitrari, non solo dentro
                # docs/experiments: le cartelle intermedie vanno create o il
                # test ERRORE invece di confrontare.
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"x\n")
                if top:
                    (root / ".gitattributes").write_text(
                        "\n".join(top) + "\n", encoding="utf-8")
                if deep:
                    (root / "docs" / "experiments" / ".gitattributes").write_text(
                        "\n".join(deep) + "\n", encoding="utf-8")
                try:
                    init = subprocess.run(["git", "init", "-q"], cwd=root,
                                          capture_output=True, timeout=30,
                                          env=_git_env())
                except (OSError, subprocess.SubprocessError):   # pragma: no cover
                    self.skipTest("no git to compare against")
                if init.returncode != 0:                        # pragma: no cover
                    self.skipTest("git init failed")
                out = subprocess.run(
                    ["git", "check-attr", "text", "--", rel], cwd=root,
                    capture_output=True, text=True, timeout=30,
                    env=_git_env())
                self.assertEqual(out.returncode, 0, out.stderr)
                theirs = out.stdout.rstrip("\n").rpartition(": ")[2]
                mine = _attrs_for(root, rel, ("text",))["text"]
                self.assertEqual(
                    mine, theirs,
                    f"case {n} {top!r} + {deep!r} on {rel}: this resolver "
                    f"says {mine!r}, git says {theirs!r}")

    def test_a_redefined_binary_macro_is_not_trusted(self):
        """`[attr]binary text` inverts the pin, and git honours it.

        `binary` is the one macro this resolver expands, because it is
        built in. But git lets a repository REDEFINE it, and then
        `docs/experiments/*.txt binary` turns EOL conversion ON. Keeping
        the built-in meaning in that case reported the records pinned when
        git says the opposite -- a false green, in the file whose whole job
        is to catch exactly that. Found by review, 22 September 2026.

        This resolver cannot expand user macros, so once `binary` is
        redefined it stops treating it as the built-in and the result falls
        to unspecified: the pin test then goes RED. That is a disagreement
        with git, which is why the case is not in the agreement list, and it
        is in the safe direction.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "docs" / "experiments").mkdir(parents=True)
            (root / ".gitattributes").write_text(
                "[attr]binary text\ndocs/experiments/*.txt binary\n",
                encoding="utf-8")
            self.assertEqual(
                _attrs_for(root, "docs/experiments/a.txt", ("text",))["text"],
                "unspecified",
                "a redefined `binary` was still expanded as the built-in, so "
                "a tree git treats as UNPINNED was reported as pinned")

    def test_macro_definition_is_not_read_as_a_pattern(self):
        """`[attr]myrec` defines a macro; it is not a glob.

        Read as a pattern it is a character class -- one of a, t, r --
        followed by the literal `myrec`, so it would match a file called
        `amyrec` and set that file's attributes from the macro's body. The
        skip that prevents this shipped without a test: replacing it with
        `if False:` left the whole suite green, which is verbatim the defect
        this PR charges elsewhere. Found by review, 22 September 2026.

        This resolver does not EXPAND user macros either. That is documented
        where the skip lives, and it fails safe: a file pinned only through a
        macro reads unspecified here, so the pin test goes red rather than
        quietly passing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "docs" / "experiments").mkdir(parents=True)
            (root / ".gitattributes").write_text(
                "[attr]myrec -text\n", encoding="utf-8")
            for rel in ("amyrec", "docs/experiments/amyrec"):
                self.assertEqual(
                    _attrs_for(root, rel, ("text",))["text"], "unspecified",
                    f"the macro DEFINITION line matched {rel} as if it were a "
                    "glob with a character class")

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
