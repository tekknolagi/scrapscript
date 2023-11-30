range 4
. range =
  | 1 -> [0]
  | i -> range (i - 1) +< (i - 1)