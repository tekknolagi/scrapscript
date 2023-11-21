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

## Run Tests

```bash
pytest tests
```

## Questions

- List indexing
  - Should it be supported?
  - We have cons. Do we want car/cdr?
- Pattern matching
  - Sure you want just matching?
  - No ternary?
- Alternatives
  - Has the syntax changed to hashes?
  - Hashes work pretty well since alternatives are like parameterized tags
- Records
  - Has the syntax changed to colons?
  - Equals was nice to reuse Assign parsing
- Where
  - Comma syntax stops being useful when you want 3 levels of wheres
- Live repl?
