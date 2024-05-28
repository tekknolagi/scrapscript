import unittest
import subprocess

from compiler import compile_to_binary


class CompilerEndToEndTests(unittest.TestCase):
    def _run(self, code: str) -> str:
        binary = compile_to_binary(code, memory=1024, debug=False)
        result = subprocess.run(binary, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout

    def test_int(self) -> None:
        self.assertEqual(self._run("1"), "1\n")

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

    def test_record_access(self) -> None:
        self.assertEqual(self._run("rec@a . rec = {a = 1, b = 2}"), "1\n")

    def test_hole(self) -> None:
        self.assertEqual(self._run("()"), "()\n")

    def test_variant(self) -> None:
        self.assertEqual(self._run("# foo 123"), "#foo 123\n")

    def test_function(self) -> None:
        self.assertEqual(self._run("f 1 . f = x -> x + 1"), "2\n")

    def test_match_int(self) -> None:
        self.assertEqual(self._run("f 3 . f = | 1 -> 2 | 3 -> 4"), "4\n")

    def test_match_list(self) -> None:
        self.assertEqual(self._run("f [4, 5] . f = | [1, 2] -> 3 | [4, 5] -> 6"), "6\n")

    def test_match_record(self) -> None:
        self.assertEqual(self._run("f {a = 4, b = 5} . f = | {a = 1, b = 2} -> 3 | {a = 4, b = 5} -> 6"), "6\n")


if __name__ == "__main__":
    unittest.main()
