# Scrapscript Interpreter

See [scrapscript.org](https://scrapscript.org/) for some more information. Keep
in mind that the syntax on the website will change a little bit in the coming
weeks to match this repository.

Take a look inside [scrapscript.py](scrapscript.py) and all of its tests to get
an idea for how the language works.

## Usage

We support python3.8+.

```bash
# With a file via STDIN
python3 scrapscript.py eval < examples/0_home/factorial.scrap

# With a string literal
python3 scrapscript.py eval "1 + 2"

# With multiple inputs
echo "4" | python3 scrapscript.py apply "n -> n + 1"

# With a REPL
python3 scrapscript.py repl
```

or with [Cosmopolitan](https://justine.lol/cosmopolitan/index.html):

```bash
./build-com

# With a file via STDIN
./scrapscript.com eval < examples/0_home/factorial.scrap

# With a string literal
./scrapscript.com eval "1 + 2"

# With multiple inputs
echo "4" | ./scrapscript.com apply "n -> n + 1"

# With a REPL
./scrapscript.com repl
```

(if you have an exec format error and use Zsh, either upgrade Zsh or prefix with `sh`)

or with Docker:

```bash
# With a file via STDIN (mount your local directory)
docker run --mount type=bind,source="$(pwd)",target=/mnt -i -t ghcr.io/tekknolagi/scrapscript:trunk eval < /mnt/examples/0_home/factorial.scrap

# With a string literal
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk eval "1 + 2"

# With multiple inputs
echo "4" | docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk eval "n -> n + 1"

# With a REPL
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk repl
```

## Running Tests

```bash
python3 scrapscript.py test
```
