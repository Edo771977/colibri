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

Comments and string literals are stripped before matching, because the
anti-pattern gets quoted while being described: this docstring, the commit
message and the comment above `reserve_graph` all contain the text
`reserve(&ctx->y, ...)`, and a checker that reads prose would fail on its own
documentation. `#if 0` blocks are NOT stripped -- that needs a preprocessor,
and dead code carrying the anti-pattern should be fixed or deleted anyway.
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
# An optional `(void**)` cast: `reserve_bytes((void**)&ctx->qx, ...)` is the
# idiom already used in this file, so a graph buffer could pick it up.
_TARGET = r"\(\s*(?:\(\s*void\s*\*\*\s*\)\s*)?&\s*\w+\s*->\s*" + _FIELD
_SUFFIX = r"(?:_bytes|_pinned|_pinned_bytes)?"

BARE = re.compile(r"\breserve" + _SUFFIX + r"\s*" + _TARGET)
WRAPPED = re.compile(r"\breserve_graph" + _SUFFIX + r"\s*\(\s*\w+\s*,\s*"
                     r"(?:\(\s*void\s*\*\*\s*\)\s*)?&\s*\w+\s*->\s*" + _FIELD)


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
        self.code = strip_comments_and_strings(
            SOURCE.read_text(encoding="utf-8", errors="replace"))

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
