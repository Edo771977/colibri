"""The graph-buffer invariant is checked here because the compiler will not.

`reserve_graph*` exists so that growing one of the eight buffers a captured
CUDA graph baked addresses into also invalidates that graph. The commit that
introduced the wrappers claimed a tenth call site "cannot forget, because
forgetting means calling reserve() on a field whose type says which wrapper
it needs". A review showed the type says nothing: `ctx->y` is a `float*`
exactly like `ctx->ac`, `ctx->aq`, `ctx->al` and `ctx->pipe_buf[slot]`, plain
`reserve()` stays in the file and stays correct on those other fields, and a
tenth site writing `reserve(&ctx->y, ...)` would compile without a word.

So the convention is enforced here instead. This is a source-text check, not
a build check, which is the point: `make check` runs it on every platform,
including the ones with no CUDA toolchain, and a new call site is caught in
review rather than on the one machine that has a card. What it CANNOT catch
is the invariant being gutted at the other end -- emptying `graph_bufs_moved`
leaves every call site spelled correctly and this file green. That half is
`tests/test_grouped_g4_cuda.cu`, which needs a GPU.

Comments, string literals and `#if 0` blocks are stripped before matching.
Comments and strings because the anti-pattern gets quoted while being
described: this docstring, the commit message and the comment above
`reserve_graph` all contain the text `reserve(&ctx->y, ...)`, and a checker
that reads prose would fail on its own documentation. `#if 0` because a
review showed the second test below could be satisfied by a dead block
naming a wrapper whose live call site had been deleted -- the guard against
the first test passing vacuously was itself satisfiable by code that does
not exist.

WHAT THIS DOES NOT CATCH, stated because the alternative is a comment that
promises more than it delivers:
  - a NINTH buffer. GRAPH_FIELDS is a closed list; adding a field to
    DeviceContext that a capture bakes in, and reserving it bare, passes.
    This catches the eight being renamed or their call sites regressing,
    not the set growing.
  - `graph_bufs_moved` being gutted. Every call site stays spelled right
    and this file stays green; that half is tests/test_grouped_g4_cuda.cu,
    which needs a card.
  - the wrong VARIANT of wrapper (pinned where device was meant). Both
    tests pass; only the types would catch it, and they do not.
"""

import pathlib
import re
import unittest

# The eight fields a captured graph can hold an address of: the five device
# buffers the capturable branches (3 and 4) touch, plus the three pinned host
# buffers whose addresses go into the recorded async memcpys. No other struct
# in backend_cuda.cu declares a field by any of these names, which is why the
# patterns below can accept any `->` base rather than just ctx/dc: a review
# found three more DeviceContext locals (`home`, `src`, `c`) in the
# multi-device helpers, and pinning the base name to two of them left a
# tenth site able to pass in silence.
GRAPH_FIELDS = ("x", "y", "gate", "up", "group_desc", "host_x", "host_y", "host_desc")

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "backend_cuda.cu"

_FIELD = "(" + "|".join(GRAPH_FIELDS) + r")\b"
# An optional cast: `reserve_bytes((void**)&ctx->qx, ...)` is the idiom
# already used in this file, so a graph buffer could pick it up.
_CAST = r"(?:\(\s*(?:void|float)\s*\*\*\s*\)\s*)?"
# The base can be indexed and can chain, and the member can be reached with
# `.` as well as `->`: `&g_ctx[i].y` is how this file already writes a
# per-device loop (g_ctx[i].device, g_ctx[i].tensor_count), so a tenth site
# in one would naturally look like that -- and pinning the accessor to `->`
# let it through. An optional paren covers `&(dc->y)`.
_BASE = r"\w+\s*(?:\[[^\]]*\]\s*)?(?:->|\.)\s*"
_TARGET = r"\(\s*" + _CAST + r"\(?\s*&\s*\(?\s*(?:" + _BASE + r")+" + _FIELD
_SUFFIX = r"(?:_bytes|_pinned|_pinned_bytes)?"

BARE = re.compile(r"\breserve" + _SUFFIX + r"\s*" + _TARGET)
WRAPPED = re.compile(r"\breserve_graph" + _SUFFIX + r"\s*\(\s*\w+\s*,\s*"
                     + _CAST + r"\(?\s*&\s*\(?\s*(?:" + _BASE + r")+" + _FIELD)


def strip_if_zero(text):
    """Blank out `#if 0` blocks, keeping every byte offset intact.

    Not a preprocessor: it tracks nesting so an `#ifdef` inside a dead block
    does not end it early, and stops at the `#else` that makes the rest live.
    That is enough for dead code, which is all this needs to see.
    """
    out = list(text)
    depth = 0          # nesting inside the dead region, 0 = not in one
    lines = text.splitlines(keepends=True)
    pos = 0
    for line in lines:
        bare = line.strip()
        if depth == 0:
            if re.match(r"#\s*if\s+0\s*$", bare):
                depth = 1
                for k in range(pos, pos + len(line)):
                    if out[k] != "\n":
                        out[k] = " "
        else:
            if re.match(r"#\s*if", bare):
                depth += 1
            elif re.match(r"#\s*endif", bare):
                depth -= 1
            elif depth == 1 and re.match(r"#\s*el(se|if)", bare):
                depth = 0          # the rest of the block is live
            for k in range(pos, pos + len(line)):
                if out[k] != "\n":
                    out[k] = " "
        pos += len(line)
    return "".join(out)


def strip_comments_and_strings(text):
    """Blank out comments and literals, keeping every byte offset intact.

    Offsets are preserved so a match still reports the right line. Newlines
    survive inside block comments for the same reason.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            while i < n and i < n and text[i : i + 2] == "*/":
                out[i] = out[i + 1] = " "
                i += 2
                break
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n:
                        # Not the newline: a line continuation inside a
                        # literal must keep its newline or every line after
                        # it is reported one short.
                        if text[i] != "\n":
                            out[i] = " "
                        i += 1
                    continue
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


class GraphBufferReserves(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        # Not a skip: an absent source is the failure mode the second test
        # below exists to catch, and a skip would report it as green.
        self.assertTrue(SOURCE.is_file(), f"{SOURCE} not found from the test directory")
        self.code = strip_if_zero(strip_comments_and_strings(
            SOURCE.read_text(encoding="utf-8", errors="replace")))

    def _line(self, offset):
        return self.code.count("\n", 0, offset) + 1

    def test_no_graph_buffer_is_reserved_without_invalidating(self):
        """A bare reserve() on one of the eight is the bug, in one grep."""
        offenders = [
            f"backend_cuda.cu:{self._line(m.start())}: {' '.join(m.group(0).split())}"
            for m in BARE.finditer(self.code)
        ]
        self.assertEqual(
            offenders,
            [],
            "these call sites move a buffer a captured graph may hold the "
            "address of, without invalidating it -- use the matching "
            "reserve_graph* wrapper:\n  " + "\n  ".join(offenders),
        )

    def test_every_graph_field_is_still_reserved_through_a_wrapper(self):
        """Guards against passing because the call sites went away.

        Matched against the stripped source, so a comment naming a wrapper
        cannot satisfy it -- which it could while this ran over raw text.
        """
        seen = set(WRAPPED.findall(self.code))
        missing = sorted(set(GRAPH_FIELDS) - seen)
        self.assertEqual(
            missing,
            [],
            "no reserve_graph* call reserves these fields any more; either "
            "the buffers were renamed and this list is stale, or the "
            "invalidation was dropped: " + ", ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
