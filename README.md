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
python3 scrapscript.py eval < examples/0_home/factorial.scrap

# With a string literal
python3 scrapscript.py eval <<< '1 + 2'

# With a REPL
python3 scrapscript.py repl
```

or with [Cosmopolitan](https://justine.lol/cosmopolitan/index.html):

```bash
./util/build-com

# With a file
./scrapscript.com eval < examples/0_home/factorial.scrap

# With a string literal
./scrapscript.com eval <<< '1 + 2'

# With a REPL
./scrapscript.com repl
```

(if you have an exec format error and use Zsh, either upgrade Zsh or prefix
with `sh`)

or with Docker:

```bash
# With a file (mount your local directory)
docker run --mount type=bind,source="$(pwd)",target=/mnt -i -t ghcr.io/tekknolagi/scrapscript:trunk eval < /mnt/examples/0_home/factorial.scrap

# With a string literal
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk eval <<< '1 + 2'

# With a REPL
docker run -i -t ghcr.io/tekknolagi/scrapscript:trunk repl
```

### The experimental compiler:

#### Normal ELF

```bash
./scrapscript.py compile some.scrap  # produces output.c
./scrapscript.py compile some.scrap --compile  # produces a.out
```

#### Cosmopolitan

```bash
CC=~/Downloads/cosmos/bin/cosmocc ./scrapscript.py compile some.scrap  --compile # produces a.out
```

#### Wasm

```bash
CC=/opt/wasi-sdk/bin/clang \
CFLAGS=-D_WASI_EMULATED_MMAN \
LDFLAGS=-lwasi-emulated-mman \
./scrapscript.py compile some.scrap --compile  # produces a.out
```

## Development Workflow

### Running Tests

```bash
python scrapscript_tests.py
```

### Type Checking the Python Sources
```bash
mypy --strict scrapscript.py
```

### Formatting the Python Sources
```bash
ruff format scrapscript.py
```

### Checking for Format Errors
```bash
ruff check scrapscript.py
```

### Using `uv`
If you choose to use `uv` to manage development dependencies, you can run any of the previous four commands by prefixing them with `uv run`, e.g.:

```bash
uv run python scrapscript_tests.py
```
