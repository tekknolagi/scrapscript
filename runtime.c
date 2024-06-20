#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef STATIC_HEAP
#include <sys/mman.h>
#endif

#define ALWAYS_INLINE inline __attribute__((always_inline))
#define NEVER_INLINE __attribute__((noinline))

const int kPointerSize = sizeof(void*);
typedef intptr_t word;
typedef uintptr_t uword;

// Garbage collector core by Andy Wingo <wingo@pobox.com>.

struct gc_obj {
  uintptr_t tag;  // low bit is 0 if forwarding ptr
};

// The low bit of the pointer is 1 if it's a heap object and 0 if it's an
// immediate integer
struct object;

bool is_small_int(struct object* obj) {
  return (((uword)obj) & kSmallIntTagMask) == kSmallIntTag;
}
bool is_immediate_not_small_int(struct object* obj) {
  return (((uword)obj) & (kPrimaryTagMask & ~kSmallIntTagMask)) != 0;
}
bool is_heap_object(struct object* obj) {
  return (((uword)obj) & kPrimaryTagMask) == kHeapObjectTag;
}
struct object* empty_list() { return (struct object*)kEmptyListTag; }
bool is_empty_list(struct object* obj) { return obj == empty_list(); }
struct object* hole() { return (struct object*)kHoleTag; }
bool is_hole(struct object* obj) {
  return (((uword)obj) & kImmediateTagMask) == kHoleTag;
}
static ALWAYS_INLINE bool is_small_string(struct object* obj) {
  return (((uword)obj) & kImmediateTagMask) == kSmallStringTag;
}
static ALWAYS_INLINE uword small_string_length(struct object* obj) {
  assert(is_small_string(obj));
  return (((uword)obj) >> kImmediateTagBits) & kMaxSmallStringLength;
}
static ALWAYS_INLINE struct object* mksmallstring(const char* data,
                                                  uword length) {
  assert(length <= kMaxSmallStringLength);
  uword result = 0;
  for (word i = length - 1; i >= 0; i--) {
    result = (result << kBitsPerByte) | data[i];
  }
  struct object* result_obj =
      (struct object*)((result << kBitsPerByte) |
                       (length << kImmediateTagBits) | kSmallStringTag);
  assert(!is_heap_object(result_obj));
  assert(is_small_string(result_obj));
  assert(small_string_length(result_obj) == length);
  return result_obj;
}
bool is_empty_string(struct object* obj) { return obj == mksmallstring("", 0); }
static ALWAYS_INLINE char small_string_at(struct object* obj, uword index) {
  assert(is_small_string(obj));
  assert(index < small_string_length(obj));
  // +1 for (length | tag) byte
  return ((uword)obj >> ((index + 1) * kBitsPerByte)) & 0xFF;
}
struct gc_obj* as_heap_object(struct object* obj) {
  assert(is_heap_object(obj));
  assert(kHeapObjectTag == 1);
  return (struct gc_obj*)((uword)obj - 1);
}

static const uintptr_t kNotForwardedBit = 1ULL;
int is_forwarded(struct gc_obj* obj) {
  return (obj->tag & kNotForwardedBit) == 0;
}
struct gc_obj* forwarded(struct gc_obj* obj) {
  assert(is_forwarded(obj));
  return (struct gc_obj*)obj->tag;
}
void forward(struct gc_obj* from, struct gc_obj* to) {
  assert(!is_forwarded(from));
  assert((((uintptr_t)to) & kNotForwardedBit) == 0);
  from->tag = (uintptr_t)to;
}

struct gc_heap;

typedef void (*VisitFn)(struct object**, struct gc_heap*);

// To implement by the user:
size_t heap_object_size(struct gc_obj* obj);
size_t trace_heap_object(struct gc_obj* obj, struct gc_heap* heap,
                         VisitFn visit);
void trace_roots(struct gc_heap* heap, VisitFn visit);

struct space {
  uintptr_t start;
  uintptr_t size;
};

struct gc_heap {
  uintptr_t hp;
  uintptr_t limit;
  uintptr_t from_space;
  uintptr_t to_space;
  size_t size;
};

static uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}
static uintptr_t align_size(uintptr_t size) {
  return align(size, sizeof(uintptr_t));
}

#ifdef STATIC_HEAP
#define make_space(mem, sz)                                                    \
  (struct space) { .start = (uintptr_t)mem, .size = sz }
void init_heap(struct gc_heap* heap, struct space space) {
  if (align(space.size, kPageSize) != space.size) {
    fprintf(stderr, "static heap size (%lu) must be a multiple of %lu\n",
            space.size, kPageSize);
    abort();
  }
  heap->size = space.size;
  heap->to_space = heap->hp = space.start;
  heap->from_space = heap->limit = heap->hp + space.size / 2;
}
void destroy_space(struct space space) {}
#else
struct space make_space(uintptr_t size) {
  size = align(size, kPageSize);
  void* mem = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    abort();
  }
  return (struct space){(uintptr_t)mem, size};
}
void init_heap(struct gc_heap* heap, struct space space) {
  heap->size = space.size;
  heap->to_space = heap->hp = space.start;
  heap->from_space = heap->limit = heap->hp + space.size / 2;
}
void destroy_space(struct space space) {
  munmap((void*)space.start, space.size);
}
#endif

struct gc_obj* copy(struct gc_heap* heap, struct gc_obj* obj) {
  size_t size = heap_object_size(obj);
  struct gc_obj* new_obj = (struct gc_obj*)heap->hp;
  memcpy(new_obj, obj, size);
  forward(obj, new_obj);
  heap->hp += align_size(size);
  return new_obj;
}

void flip(struct gc_heap* heap) {
  heap->hp = heap->from_space;
  heap->from_space = heap->to_space;
  heap->to_space = heap->hp;
  heap->limit = heap->hp + heap->size / 2;
}

struct object* heap_tag(uintptr_t addr) {
  return (struct object*)(addr | (uword)1ULL);
}

#ifdef __TINYC__
// libc defines __attribute__ as an empty macro if the compiler is not GCC or
// GCC < 2. We know tcc has supported __attribute__(section(...)) for 20+ years
// so we can undefine it.
// See tinycc-devel: https://lists.nongnu.org/archive/html/tinycc-devel/2018-04/msg00008.html
// and my StackOverflow question: https://stackoverflow.com/q/78638571/569183
#undef __attribute__
#endif

extern char __start_const_heap[];
extern char __stop_const_heap[];

bool in_const_heap(struct gc_obj* obj) {
  return (uword)obj >= (uword)__start_const_heap &&
         (uword)obj < (uword)__stop_const_heap;
}

void visit_field(struct object** pointer, struct gc_heap* heap) {
  if (!is_heap_object(*pointer)) {
    return;
  }
  struct gc_obj* from = as_heap_object(*pointer);
  if (in_const_heap(from)) {
    return;
  }
  struct gc_obj* to = is_forwarded(from) ? forwarded(from) : copy(heap, from);
  *pointer = heap_tag((uintptr_t)to);
}

void collect(struct gc_heap* heap) {
  flip(heap);
  uintptr_t scan = heap->hp;
  trace_roots(heap, visit_field);
  while (scan < heap->hp) {
    struct gc_obj* obj = (struct gc_obj*)scan;
    scan += align_size(trace_heap_object(obj, heap, visit_field));
  }
#ifndef NDEBUG
  // Zero out the rest of the heap for debugging
  memset((void*)scan, 0, heap->limit - scan);
#endif
}

#if defined(__builtin_expect)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) x
#define UNLIKELY(x) x
#endif
#define ALLOCATOR __attribute__((__malloc__))

static NEVER_INLINE ALLOCATOR struct object* allocate_slow_path(
    struct gc_heap* heap, size_t size) {
  // size is already aligned
  collect(heap);
  if (UNLIKELY(heap->limit - heap->hp < size)) {
    fprintf(stderr, "out of memory\n");
    abort();
  }
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  heap->hp = new_hp;
  return heap_tag(addr);
}

static ALWAYS_INLINE ALLOCATOR struct object* allocate(struct gc_heap* heap,
                                                       size_t size) {
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  if (UNLIKELY(heap->limit < new_hp)) {
    return allocate_slow_path(heap, size);
  }
  heap->hp = new_hp;
  return heap_tag(addr);
}

// Application

#define FOREACH_TAG(TAG)                                                       \
  TAG(TAG_LIST)                                                                \
  TAG(TAG_CLOSURE)                                                             \
  TAG(TAG_RECORD)                                                              \
  TAG(TAG_STRING)                                                              \
  TAG(TAG_VARIANT)

enum {
// All odd becase of the kNotForwardedBit
#define ENUM_TAG(TAG) TAG = __COUNTER__ * 2 + 1,
  FOREACH_TAG(ENUM_TAG)
#undef ENUM_TAG
};

struct list {
  struct gc_obj HEAD;
  struct object* first;
  struct object* rest;
};

typedef struct object* (*ClosureFn)(struct object*, struct object*);

// TODO(max): Figure out if there is a way to do a PyObject_HEAD version of
// this where each closure actually has its own struct with named members
struct closure {
  struct gc_obj HEAD;
  ClosureFn fn;
  size_t size;
  struct object* env[];
};

struct record_field {
  size_t key;
  struct object* value;
};

struct record {
  struct gc_obj HEAD;
  size_t size;
  struct record_field fields[];
};

struct heap_string {
  struct gc_obj HEAD;
  size_t size;
  char data[];
};

struct variant {
  struct gc_obj HEAD;
  size_t tag;
  struct object* value;
};

size_t variable_size(size_t base, size_t count) {
  return base + count * kPointerSize;
}

size_t record_size(size_t count) {
  return sizeof(struct record) + count * sizeof(struct record_field);
}

size_t heap_string_size(size_t count) {
  return sizeof(struct heap_string) + count;
}

size_t heap_object_size(struct gc_obj* obj) {
  switch (obj->tag) {
    case TAG_LIST:
      return sizeof(struct list);
    case TAG_CLOSURE:
      return variable_size(sizeof(struct closure),
                           ((struct closure*)obj)->size);
    case TAG_RECORD:
      return record_size(((struct record*)obj)->size);
    case TAG_STRING:
      return heap_string_size(((struct heap_string*)obj)->size);
    case TAG_VARIANT:
      return sizeof(struct variant);
    default:
      fprintf(stderr, "unknown tag: %lu\n", obj->tag);
      abort();
      // Pacify the C compiler
      return (size_t)-1;
  }
}

size_t trace_heap_object(struct gc_obj* obj, struct gc_heap* heap,
                         VisitFn visit) {
  switch (obj->tag) {
    case TAG_LIST:
      visit(&((struct list*)obj)->first, heap);
      visit(&((struct list*)obj)->rest, heap);
      break;
    case TAG_CLOSURE:
      for (size_t i = 0; i < ((struct closure*)obj)->size; i++) {
        visit(&((struct closure*)obj)->env[i], heap);
      }
      break;
    case TAG_RECORD:
      for (size_t i = 0; i < ((struct record*)obj)->size; i++) {
        visit(&((struct record*)obj)->fields[i].value, heap);
      }
      break;
    case TAG_STRING:
      break;
    case TAG_VARIANT:
      visit(&((struct variant*)obj)->value, heap);
      break;
    default:
      fprintf(stderr, "unknown tag: %lu\n", obj->tag);
      abort();
  }
  return heap_object_size(obj);
}

bool smallint_is_valid(word value) {
  return (value >= kSmallIntMinValue) && (value <= kSmallIntMaxValue);
}

struct object* mksmallint(word value) {
  assert(smallint_is_valid(value));
  return (struct object*)(((uword)value << kSmallIntTagBits));
}

struct object* mknum(struct gc_heap* heap, word value) {
  (void)heap;
  return mksmallint(value);
}

bool is_num(struct object* obj) { return is_small_int(obj); }

bool is_num_equal_word(struct object* obj, word value) {
  assert(smallint_is_valid(value));
  return obj == mksmallint(value);
}

word num_value(struct object* obj) {
  assert(is_num(obj));
  return ((word)obj) >> 1;  // sign extend
}

bool is_list(struct object* obj) {
  if (is_empty_list(obj)) {
    return true;
  }
  return is_heap_object(obj) && as_heap_object(obj)->tag == TAG_LIST;
}

struct list* as_list(struct object* obj) {
  assert(is_list(obj));
  return (struct list*)as_heap_object(obj);
}

struct object* list_first(struct object* obj) {
  assert(!is_empty_list(obj));
  return as_list(obj)->first;
}

struct object* list_rest(struct object* list) {
  assert(!is_empty_list(list));
  return as_list(list)->rest;
}

struct object* mklist(struct gc_heap* heap) {
  struct object* result = allocate(heap, sizeof(struct list));
  as_heap_object(result)->tag = TAG_LIST;
  as_list(result)->first = empty_list();
  as_list(result)->rest = empty_list();
  return result;
}

bool is_closure(struct object* obj) {
  return is_heap_object(obj) && as_heap_object(obj)->tag == TAG_CLOSURE;
}

struct closure* as_closure(struct object* obj) {
  assert(is_closure(obj));
  return (struct closure*)as_heap_object(obj);
}

struct object* mkclosure(struct gc_heap* heap, ClosureFn fn, size_t size) {
  struct object* result =
      allocate(heap, variable_size(sizeof(struct closure), size));
  as_heap_object(result)->tag = TAG_CLOSURE;
  as_closure(result)->fn = fn;
  as_closure(result)->size = size;
  // Assumes the items will be filled in immediately after calling mkclosure so
  // they are not initialized
  return result;
}

ClosureFn closure_fn(struct object* obj) { return as_closure(obj)->fn; }

void closure_set(struct object* closure, size_t i, struct object* item) {
  struct closure* c = as_closure(closure);
  assert(i < c->size);
  c->env[i] = item;
}

struct object* closure_get(struct object* closure, size_t i) {
  struct closure* c = as_closure(closure);
  assert(i < c->size);
  return c->env[i];
}

struct object* closure_call(struct object* closure, struct object* arg) {
  ClosureFn fn = closure_fn(closure);
  return fn(closure, arg);
}

bool is_record(struct object* obj) {
  return is_heap_object(obj) && as_heap_object(obj)->tag == TAG_RECORD;
}

struct record* as_record(struct object* obj) {
  assert(is_record(obj));
  return (struct record*)as_heap_object(obj);
}

struct object* mkrecord(struct gc_heap* heap, size_t size) {
  // size is the number of fields, each of which has an index and a value
  // (object)
  struct object* result = allocate(heap, record_size(size));
  as_heap_object(result)->tag = TAG_RECORD;
  as_record(result)->size = size;
  // Assumes the items will be filled in immediately after calling mkrecord so
  // they are not initialized
  return result;
}

void record_set(struct object* record, size_t index,
                struct record_field field) {
  struct record* r = as_record(record);
  assert(index < r->size);
  r->fields[index] = field;
}

struct object* record_get(struct object* record, size_t key) {
  struct record* r = as_record(record);
  struct record_field* fields = r->fields;
  for (size_t i = 0; i < r->size; i++) {
    struct record_field field = fields[i];
    if (field.key == key) {
      return field.value;
    }
  }
  return NULL;
}

bool is_string(struct object* obj) {
  if (is_small_string(obj)) {
    return true;
  }
  return is_heap_object(obj) && as_heap_object(obj)->tag == TAG_STRING;
}

struct heap_string* as_heap_string(struct object* obj) {
  assert(is_string(obj));
  return (struct heap_string*)as_heap_object(obj);
}

struct object* mkstring_uninit_private(struct gc_heap* heap, size_t size) {
  assert(size > kMaxSmallStringLength);  // can't fill in small string later
  struct object* result = allocate(heap, heap_string_size(size));
  as_heap_object(result)->tag = TAG_STRING;
  as_heap_string(result)->size = size;
  return result;
}

struct object* mkstring(struct gc_heap* heap, const char* data, uword length) {
  if (length <= kMaxSmallStringLength) {
    return mksmallstring(data, length);
  }
  struct object* result = mkstring_uninit_private(heap, length);
  memcpy(as_heap_string(result)->data, data, length);
  return result;
}

static ALWAYS_INLINE uword string_length(struct object* obj) {
  if (is_small_string(obj)) {
    return small_string_length(obj);
  }
  return as_heap_string(obj)->size;
}

char string_at(struct object* obj, uword index) {
  if (is_small_string(obj)) {
    return small_string_at(obj, index);
  }
  return as_heap_string(obj)->data[index];
}

bool is_variant(struct object* obj) {
  return is_heap_object(obj) && as_heap_object(obj)->tag == TAG_VARIANT;
}

struct variant* as_variant(struct object* obj) {
  assert(is_variant(obj));
  return (struct variant*)as_heap_object(obj);
}

struct object* mkvariant(struct gc_heap* heap, size_t tag) {
  struct object* result = allocate(heap, sizeof(struct variant));
  as_heap_object(result)->tag = TAG_VARIANT;
  as_variant(result)->tag = tag;
  return result;
}

size_t variant_tag(struct object* obj) { return as_variant(obj)->tag; }

struct object* variant_value(struct object* obj) {
  return as_variant(obj)->value;
}

void variant_set(struct object* variant, struct object* value) {
  as_variant(variant)->value = value;
}

#define MAX_HANDLES 1024

struct handle_scope {
  struct object*** base;
};

static struct object** handle_stack[MAX_HANDLES];
static struct object*** handles = handle_stack;
static struct object*** handles_end = &handle_stack[MAX_HANDLES];

void pop_handles(void* local_handles) {
  handles = ((struct handle_scope*)local_handles)->base;
}

#define HANDLES()                                                              \
  struct handle_scope local_handles __attribute__((__cleanup__(pop_handles))); \
  local_handles.base = handles;
#define GC_PROTECT(x)                                                          \
  assert(handles != handles_end);                                              \
  (*handles++) = (struct object**)(&x)
#define GC_HANDLE(type, name, val)                                             \
  type name = val;                                                             \
  GC_PROTECT(name)

void trace_roots(struct gc_heap* heap, VisitFn visit) {
  for (struct object*** h = handle_stack; h != handles; h++) {
    visit(*h, heap);
  }
}

struct gc_heap heap_object;
struct gc_heap* heap = &heap_object;

struct object* num_add(struct object* a, struct object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) + num_value(b));
}

struct object* num_sub(struct object* a, struct object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) - num_value(b));
}

struct object* num_mul(struct object* a, struct object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) * num_value(b));
}

struct object* list_cons(struct object* item, struct object* list) {
  HANDLES();
  GC_PROTECT(item);
  GC_PROTECT(list);
  struct object* result = mklist(heap);
  as_list(result)->first = item;
  as_list(result)->rest = list;
  return result;
}

struct object* heap_string_concat(struct object* a, struct object* b) {
  uword a_size = string_length(a);
  uword b_size = string_length(b);
  assert(a_size + b_size > kMaxSmallStringLength);
  HANDLES();
  GC_PROTECT(a);
  GC_PROTECT(b);
  struct object* result = mkstring_uninit_private(heap, a_size + b_size);
  for (uword i = 0; i < a_size; i++) {
    as_heap_string(result)->data[i] = string_at(a, i);
  }
  for (uword i = 0; i < b_size; i++) {
    as_heap_string(result)->data[a_size + i] = string_at(b, i);
  }
  return result;
}

static ALWAYS_INLINE struct object* small_string_concat(struct object* a,
                                                        struct object* b) {
  uword a_size = string_length(a);
  uword b_size = string_length(b);
  assert(a_size + b_size <= kMaxSmallStringLength);
  char data[kMaxSmallStringLength];
  for (uword i = 0; i < a_size; i++) {
    data[i] = small_string_at(a, i);
  }
  for (uword i = 0; i < b_size; i++) {
    data[a_size + i] = small_string_at(b, i);
  }
  return mksmallstring(data, a_size + b_size);
}

ALWAYS_INLINE static struct object* string_concat(struct object* a,
                                                  struct object* b) {
  if (is_empty_string(a)) {
    return b;
  }
  if (is_empty_string(b)) {
    return a;
  }
  uword a_size = string_length(a);
  uword b_size = string_length(b);
  if (a_size + b_size <= kMaxSmallStringLength) {
    return small_string_concat(a, b);
  }
  return heap_string_concat(a, b);
}

bool string_equal_cstr_len(struct object* string, const char* cstr, uword len) {
  assert(is_string(string));
  if (string_length(string) != len) {
    return false;
  }
  for (uword i = 0; i < len; i++) {
    if (string_at(string, i) != cstr[i]) {
      return false;
    }
  }
  return true;
}

extern const char* record_keys[];
extern const char* variant_names[];

struct object* print(struct object* obj) {
  if (is_num(obj)) {
    printf("%ld", num_value(obj));
  } else if (is_list(obj)) {
    putchar('[');
    while (!is_empty_list(obj)) {
      print(list_first(obj));
      obj = list_rest(obj);
      if (!is_empty_list(obj)) {
        putchar(',');
        putchar(' ');
      }
    }
    putchar(']');
  } else if (is_record(obj)) {
    struct record* record = as_record(obj);
    putchar('{');
    for (size_t i = 0; i < record->size; i++) {
      printf("%s = ", record_keys[record->fields[i].key]);
      print(record->fields[i].value);
      if (i + 1 < record->size) {
        fputs(", ", stdout);
      }
    }
    putchar('}');
  } else if (is_closure(obj)) {
    fputs("<closure>", stdout);
  } else if (is_string(obj)) {
    putchar('"');
    for (uword i = 0; i < string_length(obj); i++) {
      putchar(string_at(obj, i));
    }
    putchar('"');
  } else if (is_variant(obj)) {
    putchar('#');
    printf("%s ", variant_names[variant_tag(obj)]);
    print(variant_value(obj));
  } else if (is_hole(obj)) {
    fputs("()", stdout);
  } else {
    assert(is_heap_object(obj));
    fprintf(stderr, "unknown tag: %lu\n", as_heap_object(obj)->tag);
    abort();
  }
  return obj;
}

struct object* println(struct object* obj) {
  print(obj);
  putchar('\n');
  return obj;
}

#ifndef MEMORY_SIZE
#define MEMORY_SIZE 4096
#endif

// Put something in the const heap so that __start_const_heap and
// __stop_const_heap are defined by the linker.
__attribute__((section("const_heap"), used)) struct heap_string private_unused_const_heap = {
    .HEAD.tag = TAG_STRING, .size = 11, .data = "hello world"
};
