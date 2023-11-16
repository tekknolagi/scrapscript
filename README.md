# Scrapscript Interpreter

## Install

```bash
poetry install
poetry shell
```

## Usage

```bash
# Evaluate a scrapscript file
scrap eval < examples/0_home/a_example.scrap

# Evaluate a scrapscript program literal
echo "1 + 2" | scrap eval
scrap apply "1 + 4"
```
