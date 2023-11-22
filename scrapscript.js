let input = `
  t1 = 123 |> await !
  t2 = 456 |> await !
  1 + f a * b
  ? 1 + f 2 == 3
  ? 1.0 + 2.0 == 3.0
  ? a == 2
  ? [ ] == f [ ]
  ? ( ) == ( )
  ? 1 >+ [ f 2 , 3 , 4 ] == [ 1 , 2 , f 3 , 4 ]
  ? ( 1 + 2 ) == ( 1 + ( 1 + 1 ) )
  ? ~~aGVsbG8gd29ybGQ= == ~~aGVsbG8gd29ybGQ=
  ? ~~64'aGVsbG8gd29ybGQ= == ~~64'aGVsbG8gd29ybGQ=
  . a : $$int = 2
  . a : $$int
  . a = 2
  . b = 4 - f c
  . c = 1
  . _ = f "hello" ++ "!"
  . f = x -> x
  . _ = h 1 2
  . h = a -> b -> a + b
  . _ = { a = 1 , b = "x" }
  . _ = { r = _ } -> ( )
  . _ = | { r = _ } -> ( )
  . _ = { r = _ , ... } -> ( )
  . { r = r , ... } = { r = 123 , .. k }
  . _ : { q : int , ... }
  . _ : { q : int }
  . _ = { a = 2 , b = "y" , .. k }
  . k = { a = 1 , b = "x" , c = 3 }
  . _ = | "a" -> 1 | "b" -> 2 | "c" -> 3 | x -> 0
  . _ = g 6
  . g = | 1 -> 1 | n -> n * g ( n - 1 )
  . greet = x -> "hello\`x\`!"
  . _ = scoop :: chocolate ( )
  . scoop : # vanilla ( ) # chocolate ( ) # strawberry ( )
  . _ = p :: point { x = 3 , y = 4 } |> # point _ -> 999
  . _ : p -> ( ) = # point _ -> ( )
  . _ : p -> ( ) = | # point _ -> ( )
  . _ : p = p :: point { x = 3 , y = 4 }
  . p : # point { x : int , y : int }
  . _ = tuple :: triplet { x = 1.0 , y = "A" , z = ~2B } |> | # pair _ -> "B" | # triplet { y = y , ... } -> y
  . _ = { z = 888 } |> { z = z , ... } -> z
  . tuple : x => y => z => # pair { x : x , y : y } # triplet { x : x , y : y , z : z }
  . _ = $123456 1 2
  . _ = $sha1'123456 1 2
  . _ = $$add 1 2
  . _ : $$int
`;

// input = `scoop :: chocolate |> | # vanilla _ -> 111 | # chocolate _ -> 222 | # strawberry _ -> 333 . scoop : # vanilla ( ) # chocolate ( ) # strawberry ( )`;

// input = `( | # vanilla _ -> 111 | # chocolate _ -> 222 | # strawberry _ -> 333 ) scoop :: chocolate . scoop : # vanilla ( ) # chocolate ( ) # strawberry ( )`;

// input = `scoop :: chocolate ( ) . scoop : # vanilla ( ) # chocolate ( ) # strawberry ( )`;

// input = `r . { r = r , ... } = { r = 123 , .. k } . k = { }`;

// input = `f { r = 234 } . f = { r = _ } -> ( )`;

// input = `f x . f = | 123 -> 456 | y -> y . x = 0`;

// input = `f x . f = | 1 -> 1 | n -> n * f ( n - 1 ) . x = 6`;

// input = `f 1 2 . f = a -> b -> a + b`;

// TODO: Make this a proper tokenizer that handles strings with blankspace.
const tokenize = x => x.replace(/ *--[^\n]*/g, '').trim().split(/[\s\n]+/g)

const tokens = tokenize(input);

const lp = n => ({pl: n, pr: n - 0.1});
const rp = n => ({pl: n, pr: n + 0.1});
const np = n => ({pl: n, pr: n + 0});
const xp = n => ({pl: n, pr: 0});
const ps = {
  "::": lp(2000),
  "": rp(1000),
  ">>": lp(14),
  "^": rp(13),
  "*": lp(12), "/": lp(12), "//": lp(12), "%": lp(12),
  "+": lp(11), "-": lp(11),
  "**": rp(10), ">*": rp(10), "++": rp(10), ">+": rp(10),
  "==": np(9), "/=": np(9), "<": np(9), ">": np(9), "<=": np(9), ">=": np(9),
  "&&": rp(8),
  "||": rp(7),
  "#": lp(5.5),
  "=>": lp(5.11),
  "->": lp(5),
  "|": rp(4.5), ":": lp(4.5), 
  "|>": lp(4.11),
  "=": rp(4),
  "!": lp(3), ".": rp(3), "?": rp(3),
  ",": xp(1), "]": xp(1), "}": xp(1),
}
function parse(ts, p = 0) {
  const x = ts.shift();
  if (x === undefined) throw new Error("unexpected end of input");
  let l; if (false) {}
  else if (x === "|") { 
    const expr = parse(ts, 5); // TODO: make this work for larger arities
    if (expr.op !== "->") throw new Error("must be function");
    l = new Fun([[expr.l, expr.r]]);
    while(ts[0] === "|") {
      ts.shift()
      const expr_ = parse(ts, 5); // TODO: make this work for larger arities
      if (expr_.op !== "->") throw new Error("must be function");
      l.branches.push([expr_.l, expr_.r]);
    } 
  }
  else if (x === "#") { 
    l = new Uni({});
    do {
      const {l:l_,op,r} = parse(ts, 6);
      if (op !== '') throw new Error(`TODO: parsing error`);
      l.types[l_.label] = r;
    } while (ts[0] === "#" && ts.shift())
  }
  else if (ps[x]) {}
  else if (x === "(") { 
    l = ts[0] === ")" ? new Hole() : parse(ts, 0); 
    ts.shift(); 
  }
  else if (x === "[") { 
    l = []; 
    if (ts[0] === "]") ts.shift(); 
    else do { 
      l.push(parse(ts, 2)); 
    } while(ts.shift() !== "]"); 
  }
  else if (x === "{") { 
    l = new Rec(); 
    if (ts[0] === "}") ts.shift(); 
    else do { 
      const {l:l_,op,r} = parse(ts, 2); 
      if (l_?.label === "...") {} // TODO
      else if (op === "=") l.data[l_.label ?? ".."] = r; 
      else if (op === "..") l.fills.push(r);
    } while(ts.shift() !== "}"); 
  }
  else if (x === "...") { l = new Var("..."); }
  else if (x === "..") { l = new Expr({}, "..", parse(ts, 2)); }
  else if (x.match(/^[0-9]+$/)) l = parseInt(x);
  else if (x.match(/^[0-9]+[0-9.]*$/)) l = parseFloat(x);
  else if (x.match(/^".+"$/)) l = JSON.parse(x);
  else if (x.match(/^[_a-z][a-z0-9]*$/)) l = new Var(x);
  else if (x.match(/^\$(sha1')?[a-z0-9]+$/)) l = new Var(x);
  else if (x.match(/^\$\$[a-z0-9]+$/)) l = new Var(x);
  else if (x.match(/^~[^~ ]+$/)) l = new Bytes([x]);
  else if (x.match(/^~~[^~ ]+$/)) l = new Bytes(x);
  else if (x.match(/^\$::[a-z]+$/)) l = new Rock(x);
  else throw new Error(`bad token: ${x}`);
  while (true) {
    let op = ts[0];
    if (!op || op === ")" || op === "]") break;
    if (!ps[op]) op = "";
    const {pl, pr} = ps[op];
    // console.log(l, op, p, pl, pr);
    if (pl < p) break;
    if (op !== "") ts.shift();
    l = new Expr(l, op, parse(ts, pr));
  }
  return l;
};

const ast = parse(tokens);

function Var(label) {this.label = label;}
function Expr(l, op, r) {this.l = l; this.op = op; this.r = r;}
function Hole() {}
function Bytes(x) {this.x = x;}
function Fun(branches, ctx = {}, xs = []) {this.branches = [...branches]; this.ctx = {...ctx}; this.xs = [...xs];}
function Rec(data = {}, fills = []) {this.data = {...data}; this.fills = [...fills];}
function Uni(types, t = null, x = null) {this.types = {...types}; this.t = t; this.x = x;}
function Rock(label) {this.label = label;}

// TODO: null matches need to bubble up
const match = (arg,x) => {
  const type = Object.getPrototypeOf(arg).constructor.name
  if (type === "Number") return arg === x ? {} : null;
  else if (type === "Var") return { [arg.label]: x };
  else if (type === "Rec") { const envs = Object.entries(arg.data).map(([k,v]) => match(v,x.data[k])); if (envs.some(x => x === null)) return null; else return Object.assign({}, ...envs); }
  else if (type === "Uni") return arg.types[x.t] ? match(arg.types[x.t],x.x) : null;
  else if (type === "Expr") if (arg.op === ":") return match(arg.l,x); else throw new Error("TODO: unexpected expression");
  // TODO: return null if no match
  else throw new Error(`TODO: match ${type}`);
};
const ops = {
  "!": (env,l,r) => eval(env, r), // TODO: skipping left eval for now
  ".": (env,l,r) => eval({ ...env, ...eval(env,r).env }, l),
  "?": (env,l,r) => {if (eval(env,r).y !== true) throw new Error(`bad assertion: ${JSON.stringify(r)}`); return eval(env, l);},
  "=": (env,l,r) => ({ env: { ...env, ...Object.fromEntries(Object.entries(match(l,r)).map(([k,v])=>[k,eval(env,v).y])) } }),
  "+": (env,l,r) => ({ y: eval(env,l).y + eval(env,r).y }),
  "-": (env,l,r) => ({ y: eval(env,l).y - eval(env,r).y }),
  "*": (env,l,r) => ({ y: eval(env,l).y * eval(env,r).y }),
  ":": (env,l,r) => ({ env: { ...env, [l.label]: eval(env,r).y } }),
  "::": (env,l,r) => ({ y: new Uni(eval(env,l).y.types, r.label) }),
  "==": (env,l,r) => ({ y: JSON.stringify(eval(env,l).y) === JSON.stringify(eval(env,r).y) }),
  ">+": (env,l,r) => ({ y: [eval(env,l).y].concat(eval(env,r).y) }),
  "++": (env,l,r) => ({ y: eval(env,l).y.concat(eval(env,r).y) }),
  "->": (env,l,r) => ({ y: new Fun([[l,r]], env) }),
  "=>": (env,l,r) => eval(env,r),
  "|>": (env,l,r) => ops[""](env,r,l),
  "": (env,l,r) => {
    const type = Object.getPrototypeOf(l).constructor.name
    if (false) {}
    else if (["Var","Expr"].includes(type))
      return ops[""](env, eval(env, l).y, r);
    else if (type === "Uni")
      return { y: new Uni(l.types, l.t, eval(env,r).y) };
    else if (type ===  "Fun") {
      l = new Fun(l.branches, l.ctx, l.xs.concat([r]));
      if (l.branches[0].length - 1 > l.xs.length) return l;
      for (const branch of l.branches) {
        const envs = l.xs.map((x,i) => match(branch[i],eval(env,x).y));
        if (envs.some(x => !x)) continue;
        return eval(Object.assign({}, env, l.ctx, ...envs), branch[branch.length - 1]);
      }
      throw new Error(`no match found`);
    } else throw new Error(`TODO: APPLY ${type} ${l} ${r} ${env}`);
  },
};
const eval = (env,x) => {
  console.log(x);
  const type = Object.getPrototypeOf(x).constructor.name
  if (false) {}
  else if (["Number","String","Boolean","Bytes","Hole","Rock"].includes(type))
    return {y:x};
  else if (type === "Var")
    if (env[x.label]) return {y:env[x.label]};
    else throw new Error(`TODO: ${x.label} not found`);
  else if (type === "Fun")
    return {y:x}; // TODO: anything else?
  else if (type === "Uni")
    return {y:x}; // TODO: eval all the sub data?
  else if (type === "Expr")
    if (!ops[x.op]) throw new Error(`TODO: op ${x.op}`);
    else return ops[x.op](env,x.l,x.r);
  else if (type === "Array")
    return {y:x.map(x_=>eval(env,x_).y)};
  else if (type === "Rec")
    return { y: new Rec(Object.fromEntries(Object.entries(x.data).map(([k,v]) => [k,eval(env,v).y])), x.fills.map(v => eval(env,v).y)) };
  else throw new Error(`TODO: EVAL ${type} ${x} ${env}`);
};

const env = {
  "$123456": eval({}, parse(tokenize("a -> b -> a + b"))).y,
  "$sha1'123456": eval({}, parse(tokenize("a -> b -> a + b"))).y,
  "$$add": eval({}, parse(tokenize("a -> b -> a + b"))).y,
  "$$int": eval({}, parse(tokenize("# int ( )"))).y,
};
console.log(eval(env,ast).y);
