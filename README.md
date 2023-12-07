# Scrapscript Interpreter

See [scrapscript.org](https://scrapscript.org/) for some more information. Keep
in mind that the syntax on the website will change a little bit in the coming
weeks to match this repository.

Take a look inside [scrapscript.py](scrapscript.py) and all of its tests to get
an idea for how the language works.

## Usage

We support python3.8+.

```bash
# With a file
python3 scrapscript.py eval examples/0_home/triangle.ss

# With a string literal
python3 scrapscript.py apply "1 + 2"

# With a REPL
python3 scrapscript.py repl
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
