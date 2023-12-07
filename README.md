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
# With a file (mount your local directory)
docker run --mount type=bind,source="$(pwd)",target=/mnt -i -t ghcr.io/tekknolagi/scrapscript:trunk eval /mnt/examples/0_home/triangle.ss

# With a string literal
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk apply "1 + 2"

# With a REPL
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk repl
```

## Running Tests

```bash
python scrapscript.py test
```
