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

or with Docker:

```bash
# Run the REPL
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk
# or
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk repl

# the rest is same as above
```

## Running Tests

```bash
python scrapscript.py test
```
