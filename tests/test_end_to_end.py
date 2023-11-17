from typing import Optional

import pytest

# pylint: disable=redefined-builtin
from scrapscript.lib.scrapscript import Env, Int, List, Object, String, eval, parse, tokenize


class TestEndToEnd:
    @pytest.mark.parametrize(
        "program, ast",
        [
            ("1", Int(1)),
            ("1 + 2", Int(3)),
            ('"abc" ++ "def"', String("abcdef")),
            ("[ ]", List([])),
            ("[ 1 , 2 ]", List([Int(1), Int(2)])),
            ("[ 1 + 2 , 3 + 4 ]", List([Int(3), Int(7)])),
        ],
    )
    def test_end_to_end(self, program: str, ast: Object) -> None:
        assert self.run(program) == ast

    def run(self, program: str, env: Optional[Env] = None) -> Object:
        tokens = tokenize(program)
        ast = parse(tokens)
        return eval(env or {}, ast)
