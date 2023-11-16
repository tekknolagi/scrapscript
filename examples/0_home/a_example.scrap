greet <| person:ron 3

. greet :: person -> text =
  | :cowboy -> "howdy"
  | :ron n -> "hi " ++ a ++ "ron" , a = text/repeat n "a"
  | :parent :m -> "hey mom"
  | :parent :f -> "greetings father"
  | :friend n -> "yo" |> list/repeat n |> string/join " "
  | :stranger "felicia" -> "bye"
  | :stranger name -> "hello " ++ name

. person =
  : cowboy
  : ron int
  : parent s , s = (: m : f)
  : friend int
  : stranger text
