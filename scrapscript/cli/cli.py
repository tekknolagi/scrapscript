import logging
import sys

import click
from click import File

from scrapscript.lib.scrapscript import parse, tokenize

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Main CLI entrypoint."""


@main.command(name="eval")
@click.argument("program-file", type=File(), default=sys.stdin)
def eval_command(
    program_file: File,
) -> None:
    program = program_file.read()  # type: ignore [attr-defined]
    tokens = tokenize(program)
    print(tokens)
    ast = parse(tokens)
    print(ast)


@main.command(name="apply")
@click.argument("program", type=str, required=True)
def apply_command(
    program: str,
) -> None:
    tokens = tokenize(program)
    print(tokens)
    ast = parse(tokens)
    print(ast)
