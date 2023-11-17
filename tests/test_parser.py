import pytest

from scrapscript.lib.scrapscript import Assign, Binop, BinopKind, Int, List, Object, ParseError, String, Var, parse


class TestParser:
    @pytest.mark.parametrize(
        "tokens, ast",
        [
            (["1"], Int(1)),
            (["123"], Int(123)),
            pytest.param(["-123"], Int(123), marks=pytest.mark.skip("TODO(max): negatives")),
            (["abc_123"], Var("abc_123")),
            (["$sha1'abc"], Var("$sha1'abc")),
            (["1", "+", "2"], Binop(BinopKind.ADD, Int(1), Int(2))),
            (["1", "+", "2", "+", "3"], Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3)))),
            (["1", "+", "2", "*", "3"], Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3)))),
            (["1", "*", "2", "+", "3"], Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3))),
            (['"abc"', "++", '"def"'], Binop(BinopKind.CONCAT, String("abc"), String("def"))),
            (["[", "]"], List([])),
            (["[", "1", ",", "2", "]"], List([Int(1), Int(2)])),
            (["a", "=", "1"], Assign(Var("a"), Int(1))),
        ],
    )
    def test_parse(self, tokens: list[str], ast: Object) -> None:
        assert parse(tokens) == ast

    @pytest.mark.parametrize(
        "tokens, message",
        [
            ([], "unexpected end of input"),
        ],
    )
    def test_parse_error(self, tokens: list[str], message: str) -> None:
        with pytest.raises(ParseError, match=message):
            parse(tokens)
