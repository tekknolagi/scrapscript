map (v -> v * 2) [1, 2, 3, 4] 4
. map = f -> xs ->
  | 0 -> []
  | i -> (map f xs (i - 1)) +< (f (xs@(i - 1)))