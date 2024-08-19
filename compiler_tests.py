import os
import unittest
import subprocess

from scrapscript import env_get_split, discover_cflags
from compiler import compile_to_string


def compile_to_binary(source: str, memory: int, debug: bool) -> str:
    import shlex
    import subprocess
    import sysconfig
    import tempfile

    cc = env_get_split("CC", shlex.split(sysconfig.get_config_var("CC")))
    cflags = discover_cflags(cc, debug)
    cflags += [f"-DMEMORY_SIZE={memory}"]
    c_code = compile_to_string(source, debug)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as c_file:
        c_file.write(c_code)
        # The platform is in the same directory as this file
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, "cli.c"), "r") as f:
            c_file.write(f.read())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as out_file:
        subprocess.run([*cc, *cflags, "-o", out_file.name, c_file.name], check=True)
    return out_file.name


class CompilerEndToEndTests(unittest.TestCase):
    def _run(self, code: str) -> str:
        use_valgrind = bool(os.environ.get("USE_VALGRIND", False))
        binary = compile_to_binary(code, memory=4096, debug=True)
        if use_valgrind:
            cmd = ["valgrind", binary]
        else:
            cmd = [binary]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout

    def test_int(self) -> None:
        self.assertEqual(self._run("1"), "1\n")

    def test_small_string(self) -> None:
        self.assertEqual(self._run('"hello"'), '"hello"\n')

    def test_small_string_concat(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), '"abcdef"\n')

    def test_heap_string(self) -> None:
        self.assertEqual(self._run('"hello world"'), '"hello world"\n')

    def test_const_list(self) -> None:
        self.assertEqual(
            self._run("""[1, "2", [3, 4], {a=1}, #foo ()]"""),
            """[1, "2", [3, 4], {a = 1}, #foo ()]\n""",
        )

    def test_add(self) -> None:
        self.assertEqual(self._run("1 + 2"), "3\n")

    def test_sub(self) -> None:
        self.assertEqual(self._run("1 - 2"), "-1\n")

    def test_mul(self) -> None:
        self.assertEqual(self._run("2 * 3"), "6\n")

    def test_list(self) -> None:
        self.assertEqual(self._run("[1, 2, 3]"), "[1, 2, 3]\n")

    def test_var(self) -> None:
        self.assertEqual(self._run("a . a = 1"), "1\n")

    def test_record(self) -> None:
        self.assertEqual(self._run("{a = 1, b = 2}"), "{a = 1, b = 2}\n")

    def test_record_builder(self) -> None:
        self.assertEqual(self._run("f 1 2 . f = x -> y -> {a = x, b = y}"), "{a = 1, b = 2}\n")

    def test_record_access(self) -> None:
        self.assertEqual(self._run("rec@a . rec = {a = 1, b = 2}"), "1\n")

    def test_record_builder_access(self) -> None:
        self.assertEqual(self._run("(f 1 2)@a . f = x -> y -> {a = x, b = y}"), "1\n")

    def test_hole(self) -> None:
        self.assertEqual(self._run("()"), "()\n")

    def test_variant(self) -> None:
        self.assertEqual(self._run("# foo 123"), "#foo 123\n")

    def test_variant_builder(self) -> None:
        self.assertEqual(self._run("f 123 . f = x -> # foo x"), "#foo 123\n")

    def test_function(self) -> None:
        self.assertEqual(self._run("f 1 . f = x -> x + 1"), "2\n")

    def test_anonymous_function_as_value(self) -> None:
        self.assertEqual(self._run("x -> x"), "<closure>\n")

    def test_anonymous_function(self) -> None:
        self.assertEqual(self._run("((x -> x + 1) 1)"), "2\n")

    def test_match_int(self) -> None:
        self.assertEqual(self._run("f 3 . f = | 1 -> 2 | 3 -> 4"), "4\n")

    def test_match_list(self) -> None:
        self.assertEqual(self._run("f [4, 5] . f = | [1, 2] -> 3 | [4, 5] -> 6"), "6\n")

    def test_match_list_spread(self) -> None:
        self.assertEqual(self._run("f [4, 5] . f = | [_, ...xs] -> xs"), "[5]\n")

    def test_match_record(self) -> None:
        self.assertEqual(self._run("f {a = 4, b = 5} . f = | {a = 1, b = 2} -> 3 | {a = 4, b = 5} -> 6"), "6\n")

    def test_match_record_too_few_keys(self) -> None:
        self.assertEqual(self._run("f {a = 4, b = 5} . f = | {a = _} -> 3 | {a = _, b = _} -> 6"), "6\n")

    @unittest.skip("TODO")
    def test_match_record_spread(self) -> None:
        self.assertEqual(self._run("f {a=1, b=2, c=3} . f = | {a=1, ...rest} -> rest"), "[5]\n")

    def test_match_hole(self) -> None:
        self.assertEqual(self._run("f () . f = | 1 -> 3 | () -> 4"), "4\n")

    def test_match_immediate_variant(self) -> None:
        self.assertEqual(self._run("f #foo () . f = | # bar 1 -> 3 | # foo () -> 4"), "4\n")

    def test_match_heap_variant(self) -> None:
        self.assertEqual(self._run("f #bar 1 . f = | # bar 1 -> 3 | # foo () -> 4"), "3\n")

    @unittest.skipIf("STATIC_HEAP" in os.environ.get("CFLAGS", ""), "Can't grow heap in static heap mode")
    def test_heap_growth(self) -> None:
        self.assertEqual(
            self._run(
                """
countdown 1000
. countdown =
| 0 -> []
| n -> n >+ countdown (n - 1)
"""
            ),
            "[" + ", ".join(str(i) for i in range(1000, 0, -1)) + "]\n",
        )


if __name__ == "__main__":
    unittest.main()
