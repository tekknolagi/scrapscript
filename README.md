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
python3 scrapscript.py eval examples/0_home/factorial.scrap

# With a string literal
python3 scrapscript.py apply "1 + 2"

# With a REPL
python3 scrapscript.py repl
```

or with [Cosmopolitan](https://justine.lol/cosmopolitan/index.html):

```bash
./build-com

# With a file
./scrapscript.com eval examples/0_home/factorial.scrap

# With a string literal
./scrapscript.com apply "1 + 2"

# With a REPL
./scrapscript.com repl
```

(if you have an exec format error and use Zsh, either upgrade Zsh or prefix
with `sh`)

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

## Running Tests

```bash
python3 scrapscript.py test
```
