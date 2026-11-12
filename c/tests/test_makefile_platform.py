import os
import shutil
import subprocess
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")


@unittest.skipUnless(MAKE, "make is required")
class MakefilePlatformTests(unittest.TestCase):
    def _dry_run(self, target, triplet, **variables):
        args = [
            MAKE,
            "--no-print-directory",
            "-B",
            "-n",
            target,
            f"TRIPLET={triplet}",
        ]
        args.extend(f"{name}={value}" for name, value in variables.items())
        return subprocess.run(
            args,
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_windows_nt_without_uname_selects_mingw_build(self):
        env = os.environ.copy()
        env["OS"] = "Windows_NT"
        env["PATH"] = ""

        result = subprocess.run(
            [MAKE, "--no-print-directory", "-B", "-n", "colibri"],
            cwd=C_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("-o colibri.exe", result.stdout)
        self.assertIn("-fopenmp", result.stdout)
        self.assertIn("-static", result.stdout)

    def test_portable_build_uses_target_architecture(self):
        cases = (
            ("x86_64-unknown-linux-gnu", "-march=x86-64-v3"),
            ("aarch64-unknown-linux-gnu", "-march=armv8-a"),
            ("powerpc64le-unknown-linux-gnu", "-mcpu=power8"),
            ("ppc64le-unknown-linux-gnu", "-mcpu=power8"),
        )

        for triplet, expected_flag in cases:
            with self.subTest(triplet=triplet):
                result = self._dry_run("portable", triplet)
                self.assertIn(expected_flag, result.stdout)

    def test_vk_links_no_vulkan_loader(self):
        # The loader is opened at run time (vk_load.h): a VK=1 binary has to
        # start where there is none, as the release archives' do. Linux takes
        # -ldl for dlopen; Windows stays fully -static (LoadLibrary is kernel32).
        cases = (
            ("x86_64-unknown-linux-gnu", "-o colibri ", ["-pthread", "-ldl"]),
            ("x86_64-w64-mingw32", "-o colibri.exe ", ["-static", "-lpsapi"]),
        )

        for triplet, output, tail in cases:
            with self.subTest(triplet=triplet):
                result = self._dry_run("colibri", triplet, VK=1)
                link = next(
                    line for line in result.stdout.splitlines() if output in line
                )
                self.assertEqual(link.split()[-len(tail):], tail)
                self.assertNotIn("-lvulkan", link)
                self.assertNotIn("-Bdynamic", link)

    def test_windows_clang_and_gcc_link_lines(self):
        # MSYS2 CLANG64 clang reports x86_64-w64-windows-gnu (no "mingw"): it
        # must still get the .exe target and the Windows libraries, link libomp
        # dynamically (no -static) and add -lpthread for libwinpthread; a MinGW
        # gcc keeps -static. Both reserve an 8 MB stack. The decision follows
        # the triple, not the compiler's name (CLANG64's gcc wrapper is clang),
        # and with a windows-gnu triple the compiler that reported it builds:
        # an environment CC is not replaced by gcc.
        cases = (
            # triple, CC in the environment (None: unset), expected compiler, clang
            ("x86_64-w64-windows-gnu", "clang", "clang", True),
            ("x86_64-w64-windows-gnu", None, "gcc", True),
            ("x86_64-w64-mingw32", None, "gcc", False),
        )

        for triplet, env_cc, expected_cc, is_clang in cases:
            with self.subTest(triplet=triplet, env_cc=env_cc):
                env = os.environ.copy()
                env.pop("CC", None)
                if env_cc is not None:
                    env["CC"] = env_cc
                result = subprocess.run(
                    [
                        MAKE,
                        "--no-print-directory",
                        "-B",
                        "-n",
                        "qwen36.exe",
                        f"TRIPLET={triplet}",
                    ],
                    cwd=C_DIR,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                link = next(
                    line
                    for line in result.stdout.splitlines()
                    if "-o qwen36.exe " in line
                )
                words = link.split()
                self.assertEqual(words[0], expected_cc)
                self.assertIn("-lpsapi", words)
                self.assertIn("-Wl,--stack,8388608", words)
                self.assertEqual("-static" in words, not is_clang)
                self.assertEqual("-lpthread" in words, is_clang)

    def test_darwin_portable_build_does_not_force_x86_architecture(self):
        missing_libomp = "/colibri-test/missing-libomp"
        result = self._dry_run(
            "portable", "arm64-apple-darwin", OMPDIR=missing_libomp
        )

        self.assertIn("clang -O3", result.stdout)
        self.assertNotIn("-mcpu=x86-64-v3", result.stdout)
        self.assertNotIn(missing_libomp, result.stdout)
        self.assertNotIn("-fopenmp", result.stdout)


if __name__ == "__main__":
    unittest.main()
