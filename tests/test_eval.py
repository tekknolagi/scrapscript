from typing import Type

import pytest

# pylint: disable=redefined-builtin
from scrapscript.lib.scrapscript import Binop, BinopKind, Env, Int, List, Object, String, Var, eval


class TestEval:
    @pytest.mark.parametrize(
        "env, ast, res",
        [
            ({}, Int(5), Int(5)),
            ({}, String("xyz"), String("xyz")),
            ({"yes": Int(123)}, Var("yes"), Int(123)),
            ({}, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)),
            ({}, Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)), Int(6)),
            ({}, Binop(BinopKind.SUB, Int(1), Int(2)), Int(-1)),
            ({}, Binop(BinopKind.MUL, Int(2), Int(3)), Int(6)),
            ({}, Binop(BinopKind.DIV, Int(2), Int(3)), Int(0)),
            ({}, Binop(BinopKind.CONCAT, String("hello"), String(" world")), String("hello world")),
        ],
    )
    def test_eval(self, env: Env, ast: Object, res: Object) -> None:
        assert eval(env, ast) == res

    @pytest.mark.parametrize(
        "env, ast, error_type, message",
        [
            ({}, Var("no"), NameError, "name 'no' is not defined"),
            ({}, Binop(BinopKind.ADD, Int(1), String("hello")), TypeError, "expected Int, got String"),
            ({}, Binop(BinopKind.CONCAT, Int(123), String(" world")), TypeError, "expected String, got Int"),
            ({}, Binop(BinopKind.CONCAT, String(" world"), Int(123)), TypeError, "expected String, got Int"),
        ],
    )
    def test_eval_error(self, env: Env, ast: Object, error_type: Type[Exception], message: str) -> None:
        with pytest.raises(error_type, match=message):
            eval(env, ast)

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        assert eval({}, exp) == List([Int(3), Int(7)])
