(f >> (x -> x) >> g) 7
. f = | 7 -> "cat"
      | 4 -> "dog"
      | _ -> "shark"
. g = | "cat" -> "kitten"
      | "dog" -> "puppy"
      |   a   -> "baby " ++ a