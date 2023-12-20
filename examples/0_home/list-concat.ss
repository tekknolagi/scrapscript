list_concatenate [1, 2, 3] [4, 5, 6]
. list_concatenate = xs ->
  | [] -> xs
  | [y, ...ys] -> list_concatenate (xs +< y) ys