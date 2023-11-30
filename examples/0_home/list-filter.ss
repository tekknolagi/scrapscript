filter (v -> v < 5) [2, 8, 4, 5, 1, 6, 3, 7] 8
. filter = f -> xs ->
  | 0 -> []
  | n -> get_rest (f xs@(n - 1))
  . get_rest =
    | 0 -> filter f xs (n - 1)
    | 1 -> filter f xs (n - 1) +< xs@(n - 1)