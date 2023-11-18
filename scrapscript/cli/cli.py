import logging
import sys

import click
from click import File

# pylint: disable=redefined-builtin
from scrapscript.lib.scrapscript import eval, parse, tokenize

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Main CLI entrypoint."""


@main.command(name="eval")
@click.argument("program-file", type=File(), default=sys.stdin)
@click.option("--debug", is_flag=True)
def eval_command(program_file: File, debug: bool) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    program = program_file.read()  # type: ignore [attr-defined]
    tokens = tokenize(program)
    print(tokens)
    ast = parse(tokens)
    print(ast)
    print(eval({}, ast))


@main.command(name="apply")
@click.argument("program", type=str, required=True)
@click.option("--debug", is_flag=True)
def apply_command(program: str, debug: bool) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    tokens = tokenize(program)
    print(tokens)
    ast = parse(tokens)
    print(ast)
    print(eval({}, ast))


if __name__ == "__main__":
    main()
