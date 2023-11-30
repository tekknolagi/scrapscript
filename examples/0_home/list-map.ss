map (v -> v * 2) [1, 2, 3, 4] 4
. map = f -> xs ->
  | 0 -> []
  | n -> (map f xs (n - 1)) +< (f (xs@(n - 1)))