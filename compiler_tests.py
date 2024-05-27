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


if __name__ == "__main__":
    unittest.main()
