import pytest

from scrapscript.lib.scrapscript import tokenize


class TestTokenizer:
    @pytest.mark.parametrize(
        "program, tokens",
        [
            ("1", ["1"]),
            ("123", ["123"]),
            ("-123", ["-123"]),
            ("1 + 2", ["1", "+", "2"]),
            ("1\n+\t2", ["1", "+", "2"]),
            ("-- 1\n2", ["2"]),
            ('"hello"', ['"hello"']),
            ('"hello world"', ['"hello world"']),
            ("[ ] ", ["[", "]"]),
            ("[ 1 , 2 ] ", ["[", "1", ",", "2", "]"]),
        ],
    )
    def test_tokenize(self, program: str, tokens: list[str]) -> None:
        assert tokenize(program) == tokens
