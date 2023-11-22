map 4 fn arr
. map =
  | 0 -> (f -> (xs -> []))
  | i -> (f -> (xs -> (map (i - 1) f xs) +< (f (xs@(i - 1)))))
. fn = v -> v * 2
. arr = [1, 2, 3, 4]