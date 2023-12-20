map (v -> v * 2) [1, 2, 3, 4]
. map = f ->
  | [] -> []
  | [x, ...xs] -> (f x) >+ (map f xs)