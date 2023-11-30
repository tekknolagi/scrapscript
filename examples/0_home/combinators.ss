Z factr 5

. S = x -> y -> z -> x -> z (y z)
. K = x -> y -> x
. I = x -> x

. A = x -> y -> y
. B = x -> y -> z -> x (y z)
. C = x -> y -> z -> x z y
. M = x -> x x
. T = x -> y -> y x
. W = x -> y -> x y y

. Y = f -> (x -> f (x x)) (x -> f (x x))
. Z = f -> (x -> f (v -> (x x) v)) (x -> f (v -> (x x) v))

. factr = facti ->
  | 0 -> 1
  | n -> (mult n) (facti (n - 1))

. double = x -> x * 2
. add = x -> y -> x + y
. mult = x -> y -> x * y
. h = x -> y -> x + (y + 1)
. plus3 = x -> x + 3
. sq = x -> x * x
. iszero = x -> x == 0