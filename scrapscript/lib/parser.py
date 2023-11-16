import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def tokenize(program: str) -> List[str]:
    # TODO: Make this a proper tokenizer that handles strings with blankspace.
    stripped = re.sub(r" *--[^\n]*", "", program).strip()
    return re.split(r"[\s\n]+", stripped)


def parse(program: str) -> None:
    tokens = tokenize(program)
    print(tokens)
