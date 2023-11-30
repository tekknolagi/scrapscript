list_concatenate [1, 2, 3] [4, 5, 6] 3
. list_concatenate = xs -> ys ->
  | 0 -> ys
  | n -> list_concatenate xs (xs@(n - 1) >+ ys) (n - 1)