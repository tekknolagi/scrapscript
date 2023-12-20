filter lt5 [2, 8, 4, 5, 1, 6, 3, 7]
. lt5 = x -> x < 5
. filter = f ->
  | [] -> []
  | [x, ...xs] -> (get_rest (f x)
    . get_rest =
      | true -> (false >+ (filter f xs))
      | false -> (2 >+ filter f xs))