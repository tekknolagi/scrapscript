# Scrapscript Interpreter

## Usage

```bash
# With a file
python scrapscript.py eval examples/0_home/triangle.ss

# With a string literal
python scrapscript.py apply "1 + 2"

# With a REPL
python scrapscript.py repl
```

### CLI

This step requires [Poetry](https://python-poetry.org/)

```bash
poetry install
poetry shell

# With a file
scrap eval < examples/0_home/triangle.ss

# With a string literal
echo "1 + 2" | scrap eval

# or
scrap apply "1 + 2"
```

## Running Tests

```bash
python scrapscript.py test
```

## Questions

- List indexing
  - Should it be supported?
  - We have cons. Do we want car/cdr?
- Alternatives
  - Has the syntax changed to hashes?
  - Hashes work pretty well since alternatives are like parameterized tags
- Records
  - Has the syntax changed to colons?
  - Equals was nice to reuse Assign parsing
- Where
  - Comma syntax stops being useful when you want 3 levels of wheres
