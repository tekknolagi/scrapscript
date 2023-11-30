map fn 4 arr
. map = f ->
  | 0 -> xs -> []
  | i -> xs -> (map f (i - 1) xs) +< (f (xs@(i - 1)))
. fn = v -> v * 2
. arr = [1, 2, 3, 4]