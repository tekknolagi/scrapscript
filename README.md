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

This step requires [Poetry](https://python-poetry.org/) when installing from source.

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
