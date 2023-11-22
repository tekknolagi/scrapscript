triangle 5 0
. triangle =
  | 0 -> (a -> a)
  | n -> (a -> triangle (n - 1) (a + n))