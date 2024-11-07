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
typedef unsigned char byte;

// Garbage collector core by Andy Wingo <wingo@pobox.com>.

typedef struct HeapObject {
  uintptr_t tag;  // low bit is 0 if forwarding ptr
} HeapObject;

// The low bit of the pointer is 1 if it's a heap object and 0 if it's an
// immediate integer
typedef struct Object Object;

bool is_small_int(Object* obj) {
  return (((uword)obj) & kSmallIntTagMask) == kSmallIntTag;
}
bool is_immediate_not_small_int(Object* obj) {
  return (((uword)obj) & (kPrimaryTagMask & ~kSmallIntTagMask)) != 0;
}
bool is_heap_object(Object* obj) {
  return (((uword)obj) & kPrimaryTagMask) == kHeapObjectTag;
}
#define empty_list() ((Object*)kEmptyListTag)
bool is_empty_list(Object* obj) { return obj == empty_list(); }
#define hole() ((Object*)kHoleTag)
bool is_hole(Object* obj) { return (uword)obj == kHoleTag; }
static ALWAYS_INLINE bool is_small_string(Object* obj) {
  return (((uword)obj) & kImmediateTagMask) == kSmallStringTag;
}
#define mk_immediate_variant(tag)                                              \
  (Object*)(((uword)(tag) << kImmediateTagBits) | kVariantTag)
static ALWAYS_INLINE bool is_immediate_variant(Object* obj) {
  return ((uword)obj & kImmediateTagMask) == kVariantTag;
}
static uword immediate_variant_tag(Object* obj) {
  assert(is_immediate_variant(obj));
  return ((uword)obj) >> kImmediateTagBits;
}
static ALWAYS_INLINE uword small_string_length(Object* obj) {
  assert(is_small_string(obj));
  return (((uword)obj) >> kImmediateTagBits) & kMaxSmallStringLength;
}
static ALWAYS_INLINE Object* mksmallstring(const char* data,
                                                  uword length) {
  assert(length <= kMaxSmallStringLength);
  uword result = 0;
  for (word i = length - 1; i >= 0; i--) {
    result = (result << kBitsPerByte) | data[i];
  }
  Object* result_obj =
      (Object*)((result << kBitsPerByte) |
                       (length << kImmediateTagBits) | kSmallStringTag);
  assert(!is_heap_object(result_obj));
  assert(is_small_string(result_obj));
  assert(small_string_length(result_obj) == length);
  return result_obj;
}
Object* empty_string() { return (Object*)kSmallStringTag; }
bool is_empty_string(Object* obj) { return obj == empty_string(); }
static ALWAYS_INLINE char small_string_at(Object* obj, uword index) {
  assert(is_small_string(obj));
  assert(index < small_string_length(obj));
  // +1 for (length | tag) byte
  return ((uword)obj >> ((index + 1) * kBitsPerByte)) & 0xFF;
}
HeapObject* as_heap_object(Object* obj) {
  assert(is_heap_object(obj));
  assert(kHeapObjectTag == 1);
  return (HeapObject*)((uword)obj - 1);
}

static const uintptr_t kNotForwardedBit = 1ULL;
int is_forwarded(HeapObject* obj) {
  return (obj->tag & kNotForwardedBit) == 0;
}
HeapObject* forwarded(HeapObject* obj) {
  assert(is_forwarded(obj));
  return (HeapObject*)obj->tag;
}
void forward(HeapObject* from, HeapObject* to) {
  assert(!is_forwarded(from));
  assert((((uintptr_t)to) & kNotForwardedBit) == 0);
  from->tag = (uintptr_t)to;
}

struct gc_heap;

typedef void (*VisitFn)(Object**, struct gc_heap*);

// To implement by the user:
size_t heap_object_size(HeapObject* obj);
size_t trace_heap_object(HeapObject* obj, struct gc_heap* heap,
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
  uintptr_t base;
  struct space space;
};

static uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}
static uintptr_t align_size(uintptr_t size) {
  return align(size, sizeof(uintptr_t));
}

#ifdef STATIC_HEAP
struct space make_space(void* mem, uintptr_t size) {
  return (struct space){(uintptr_t)mem, size};
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
void destroy_space(struct space space) {
  munmap((void*)space.start, space.size);
}
#endif

void init_heap(struct gc_heap* heap, struct space space) {
  if (align(space.size, kPageSize) != space.size) {
    fprintf(stderr, "heap size (%lu) must be a multiple of %lu\n",
            space.size, kPageSize);
    abort();
  }
  heap->space = space;
  heap->base = heap->to_space = heap->hp = space.start;
  heap->from_space = heap->limit = heap->hp + space.size / 2;
}

HeapObject* copy(struct gc_heap* heap, HeapObject* obj) {
  size_t size = heap_object_size(obj);
  HeapObject* new_obj = (HeapObject*)heap->hp;
  memcpy(new_obj, obj, size);
  forward(obj, new_obj);
  heap->hp += align_size(size);
  return new_obj;
}

void flip(struct gc_heap* heap) {
  heap->base = heap->hp = heap->from_space;
  heap->from_space = heap->to_space;
  heap->to_space = heap->hp;
  heap->limit = heap->hp + heap->space.size / 2;
}

Object* heap_tag(uintptr_t addr) {
  return (Object*)(addr | (uword)1ULL);
}

#ifdef __TINYC__
// libc defines __attribute__ as an empty macro if the compiler is not GCC or
// GCC < 2. We know tcc has supported __attribute__(section(...)) for 20+ years
// so we can undefine it.
// See tinycc-devel:
// https://lists.nongnu.org/archive/html/tinycc-devel/2018-04/msg00008.html and
// my StackOverflow question: https://stackoverflow.com/q/78638571/569183
#undef __attribute__
#endif

extern char __start_const_heap[];
extern char __stop_const_heap[];

bool in_const_heap(HeapObject* obj) {
  return (uword)obj >= (uword)__start_const_heap &&
         (uword)obj < (uword)__stop_const_heap;
}

void visit_field(Object** pointer, struct gc_heap* heap) {
  if (!is_heap_object(*pointer)) {
    return;
  }
  HeapObject* from = as_heap_object(*pointer);
  if (in_const_heap(from)) {
    return;
  }
  HeapObject* to = is_forwarded(from) ? forwarded(from) : copy(heap, from);
  *pointer = heap_tag((uintptr_t)to);
}

static bool in_heap(struct gc_heap* heap, HeapObject* obj) {
  return (uword)obj >= heap->base && (uword)obj < heap->hp;
}

void assert_in_heap(Object** pointer, struct gc_heap* heap) {
  if (!is_heap_object(*pointer)) {
    return;
  }
  HeapObject* obj = as_heap_object(*pointer);
  if (in_const_heap(obj)) {
    return;
  }
  if (!in_heap(heap, obj)) {
    fprintf(stderr, "pointer %p not in heap [%p, %p)\n", obj, (void*)heap->to_space,
            (void*)heap->hp);
    abort();
  }
}

static NEVER_INLINE void heap_verify(struct gc_heap* heap) {
  assert(heap->base <= heap->hp);
  trace_roots(heap, assert_in_heap);
  uintptr_t scan = heap->base;
  while (scan < heap->hp) {
    HeapObject* obj = (HeapObject*)scan;
    scan += align_size(trace_heap_object(obj, heap, assert_in_heap));
  }
}

void collect_no_verify(struct gc_heap* heap) {
  flip(heap);
  uintptr_t scan = heap->hp;
  trace_roots(heap, visit_field);
  while (scan < heap->hp) {
    HeapObject* obj = (HeapObject*)scan;
    scan += align_size(trace_heap_object(obj, heap, visit_field));
  }
  // TODO(max): If we have < 25% heap utilization, shrink the heap
#ifndef NDEBUG
  // Zero out the rest of the heap for debugging
  memset((void*)scan, 0, heap->limit - scan);
#endif
}

void collect(struct gc_heap* heap) {
#ifndef NDEBUG
  heap_verify(heap);
#endif
  collect_no_verify(heap);
#ifndef NDEBUG
  heap_verify(heap);
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

#ifndef STATIC_HEAP
static NEVER_INLINE void heap_grow(struct gc_heap* heap) {
  struct space old_space = heap->space;
  struct space new_space = make_space(old_space.size * 2);
#ifndef NDEBUG
  heap_verify(heap);
#endif
  init_heap(heap, new_space);
  collect_no_verify(heap);
#ifndef NDEBUG
  heap_verify(heap);
#endif
  destroy_space(old_space);
}
#endif

bool is_power_of_two(uword x) { return (x & (x - 1)) == 0; }

bool is_aligned(uword value, uword alignment) {
  assert(is_power_of_two(alignment));
  return (value & (alignment - 1)) == 0;
}

uword make_tag(uword tag, uword size_bytes) {
  assert(size_bytes <= 0xffffffff);
  return (size_bytes << kBitsPerByte) | tag;
}

byte obj_tag(HeapObject* obj) { return (obj->tag & 0xff); }

bool obj_has_tag(HeapObject* obj, byte tag) { return obj_tag(obj) == tag; }

static NEVER_INLINE void allocate_slow_path(struct gc_heap* heap, uword size) {
#ifndef STATIC_HEAP
  heap_grow(heap);
#endif
  // size is already aligned
  if (UNLIKELY(heap->limit - heap->hp < size)) {
    fprintf(stderr, "out of memory\n");
    abort();
  }
}

static ALWAYS_INLINE ALLOCATOR Object* allocate(struct gc_heap* heap,
                                                       uword tag, uword size) {
  assert(is_aligned(size, 1 << kPrimaryTagBits) && "need 3 bits for tagging");
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  if (UNLIKELY(heap->limit < new_hp)) {
    allocate_slow_path(heap, size);
    addr = heap->hp;
    new_hp = align_size(addr + size);
  }
  heap->hp = new_hp;
  ((HeapObject*)addr)->tag = make_tag(tag, size);
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
  HeapObject HEAD;
  Object* first;
  Object* rest;
};

typedef Object* (*ClosureFn)(Object*, Object*);

// TODO(max): Figure out if there is a way to do a PyObject_HEAD version of
// this where each closure actually has its own struct with named members
struct closure {
  HeapObject HEAD;
  ClosureFn fn;
  size_t size;
  Object* env[];
};

struct record_field {
  size_t key;
  Object* value;
};

struct record {
  HeapObject HEAD;
  size_t size;
  struct record_field fields[];
};

struct heap_string {
  HeapObject HEAD;
  size_t size;
  char data[];
};

struct variant {
  HeapObject HEAD;
  size_t tag;
  Object* value;
};

size_t heap_object_size(HeapObject* obj) { return obj->tag >> kBitsPerByte; }

size_t trace_heap_object(HeapObject* obj, struct gc_heap* heap,
                         VisitFn visit) {
  switch (obj_tag(obj)) {
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
      fprintf(stderr, "unknown tag: %u\n", obj_tag(obj));
      abort();
  }
  return heap_object_size(obj);
}

bool smallint_is_valid(word value) {
  return (value >= kSmallIntMinValue) && (value <= kSmallIntMaxValue);
}

#define _mksmallint(value)                                                     \
  (Object*)(((uword)(value) << kSmallIntTagBits) | kSmallIntTag)

Object* mksmallint(word value) {
  assert(smallint_is_valid(value));
  return _mksmallint(value);
}

Object* mknum(struct gc_heap* heap, word value) {
  (void)heap;
  return mksmallint(value);
}

bool is_num(Object* obj) { return is_small_int(obj); }

bool is_num_equal_word(Object* obj, word value) {
  assert(smallint_is_valid(value));
  return obj == mksmallint(value);
}

word num_value(Object* obj) {
  assert(is_num(obj));
  return ((word)obj) >> 1;  // sign extend
}

bool is_list(Object* obj) {
  if (is_empty_list(obj)) {
    return true;
  }
  return is_heap_object(obj) && obj_has_tag(as_heap_object(obj), TAG_LIST);
}

struct list* as_list(Object* obj) {
  assert(is_list(obj));
  return (struct list*)as_heap_object(obj);
}

Object* list_first(Object* obj) {
  assert(!is_empty_list(obj));
  return as_list(obj)->first;
}

Object* list_rest(Object* list) {
  assert(!is_empty_list(list));
  return as_list(list)->rest;
}

Object* mklist(struct gc_heap* heap) {
  Object* result = allocate(heap, TAG_LIST, sizeof(struct list));
  as_list(result)->first = empty_list();
  as_list(result)->rest = empty_list();
  return result;
}

bool is_closure(Object* obj) {
  return is_heap_object(obj) && obj_has_tag(as_heap_object(obj), TAG_CLOSURE);
}

struct closure* as_closure(Object* obj) {
  assert(is_closure(obj));
  return (struct closure*)as_heap_object(obj);
}

Object* mkclosure(struct gc_heap* heap, ClosureFn fn,
                         size_t num_fields) {
  Object* result = allocate(
      heap, TAG_CLOSURE, sizeof(struct closure) + num_fields * kPointerSize);
  as_closure(result)->fn = fn;
  as_closure(result)->size = num_fields;
  // Assumes the items will be filled in immediately after calling mkclosure so
  // they are not initialized
  return result;
}

ClosureFn closure_fn(Object* obj) { return as_closure(obj)->fn; }

void closure_set(Object* closure, size_t i, Object* item) {
  struct closure* c = as_closure(closure);
  assert(i < c->size);
  c->env[i] = item;
}

Object* closure_get(Object* closure, size_t i) {
  struct closure* c = as_closure(closure);
  assert(i < c->size);
  return c->env[i];
}

Object* closure_call(Object* closure, Object* arg) {
  ClosureFn fn = closure_fn(closure);
  return fn(closure, arg);
}

bool is_record(Object* obj) {
  return is_heap_object(obj) && obj_has_tag(as_heap_object(obj), TAG_RECORD);
}

struct record* as_record(Object* obj) {
  assert(is_record(obj));
  return (struct record*)as_heap_object(obj);
}

Object* mkrecord(struct gc_heap* heap, size_t num_fields) {
  Object* result = allocate(
      heap, TAG_RECORD,
      sizeof(struct record) + num_fields * sizeof(struct record_field));
  as_record(result)->size = num_fields;
  // Assumes the items will be filled in immediately after calling mkrecord so
  // they are not initialized
  return result;
}

size_t record_num_fields(Object* record) {
  return as_record(record)->size;
}

void record_set(Object* record, size_t index,
                struct record_field field) {
  struct record* r = as_record(record);
  assert(index < r->size);
  r->fields[index] = field;
}

Object* record_get(Object* record, size_t key) {
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

bool is_string(Object* obj) {
  if (is_small_string(obj)) {
    return true;
  }
  return is_heap_object(obj) && obj_has_tag(as_heap_object(obj), TAG_STRING);
}

struct heap_string* as_heap_string(Object* obj) {
  assert(is_string(obj));
  return (struct heap_string*)as_heap_object(obj);
}

Object* mkstring_uninit_private(struct gc_heap* heap, size_t count) {
  assert(count > kMaxSmallStringLength);  // can't fill in small string later
  Object* result =
      allocate(heap, TAG_STRING, sizeof(struct heap_string) + count);
  as_heap_string(result)->size = count;
  return result;
}

Object* mkstring(struct gc_heap* heap, const char* data, uword length) {
  if (length <= kMaxSmallStringLength) {
    return mksmallstring(data, length);
  }
  Object* result = mkstring_uninit_private(heap, length);
  memcpy(as_heap_string(result)->data, data, length);
  return result;
}

static ALWAYS_INLINE uword string_length(Object* obj) {
  if (is_small_string(obj)) {
    return small_string_length(obj);
  }
  return as_heap_string(obj)->size;
}

char string_at(Object* obj, uword index) {
  if (is_small_string(obj)) {
    return small_string_at(obj, index);
  }
  return as_heap_string(obj)->data[index];
}

bool is_variant(Object* obj) {
  if (is_immediate_variant(obj)) {
    return true;
  }
  return is_heap_object(obj) && obj_has_tag(as_heap_object(obj), TAG_VARIANT);
}

struct variant* as_variant(Object* obj) {
  assert(is_variant(obj));
  assert(is_heap_object(obj));  // This only makes sense for heap variants.
  return (struct variant*)as_heap_object(obj);
}

Object* mkvariant(struct gc_heap* heap, size_t tag) {
  Object* result = allocate(heap, TAG_VARIANT, sizeof(struct variant));
  as_variant(result)->tag = tag;
  return result;
}

size_t variant_tag(Object* obj) {
  if (is_immediate_variant(obj)) {
    return immediate_variant_tag(obj);
  }
  return as_variant(obj)->tag;
}

Object* variant_value(Object* obj) {
  if (is_immediate_variant(obj)) {
    return hole();
  }
  return as_variant(obj)->value;
}

void variant_set(Object* variant, Object* value) {
  as_variant(variant)->value = value;
}

#define MAX_HANDLES 4096

struct handle_scope {
  Object*** base;
};

static Object** handle_stack[MAX_HANDLES];
static Object*** handles = handle_stack;
static Object*** handles_end = &handle_stack[MAX_HANDLES];

void pop_handles(void* local_handles) {
  handles = ((struct handle_scope*)local_handles)->base;
}

#define HANDLES()                                                              \
  struct handle_scope local_handles __attribute__((__cleanup__(pop_handles))); \
  local_handles.base = handles;
#define GC_PROTECT(x)                                                          \
  assert(handles != handles_end);                                              \
  (*handles++) = (Object**)(&x)
#define GC_HANDLE(type, name, val)                                             \
  type name = val;                                                             \
  GC_PROTECT(name)

void trace_roots(struct gc_heap* heap, VisitFn visit) {
  for (Object*** h = handle_stack; h != handles; h++) {
    visit(*h, heap);
  }
}

struct gc_heap heap_object;
struct gc_heap* heap = &heap_object;

Object* num_add(Object* a, Object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) + num_value(b));
}

Object* num_sub(Object* a, Object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) - num_value(b));
}

Object* num_mul(Object* a, Object* b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a) * num_value(b));
}

Object* list_cons(Object* item, Object* list) {
  HANDLES();
  GC_PROTECT(item);
  GC_PROTECT(list);
  Object* result = mklist(heap);
  as_list(result)->first = item;
  as_list(result)->rest = list;
  return result;
}

Object* heap_string_concat(Object* a, Object* b) {
  uword a_size = string_length(a);
  uword b_size = string_length(b);
  assert(a_size + b_size > kMaxSmallStringLength);
  HANDLES();
  GC_PROTECT(a);
  GC_PROTECT(b);
  Object* result = mkstring_uninit_private(heap, a_size + b_size);
  for (uword i = 0; i < a_size; i++) {
    as_heap_string(result)->data[i] = string_at(a, i);
  }
  for (uword i = 0; i < b_size; i++) {
    as_heap_string(result)->data[a_size + i] = string_at(b, i);
  }
  return result;
}

static ALWAYS_INLINE Object* small_string_concat(Object* a_obj,
                                                        Object* b_obj) {
  // a:         CBAT
  // b:      FEDT
  // result: FEDCBAT
  assert(is_small_string(a_obj));
  assert(is_small_string(b_obj));
  uword length = small_string_length(a_obj) + small_string_length(b_obj);
  assert(length <= kMaxSmallStringLength);
  uword result = ((uword)b_obj) & ~(uword)0xFFULL;
  result <<= small_string_length(a_obj) * kBitsPerByte;
  result |= ((uword)a_obj) & ~(uword)0xFFULL;
  result |= length << kImmediateTagBits;
  result |= kSmallStringTag;
  Object* result_obj = (Object*)result;
  assert(!is_heap_object(result_obj));
  assert(is_small_string(result_obj));
  return result_obj;
}

ALWAYS_INLINE static Object* string_concat(Object* a,
                                                  Object* b) {
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

bool string_equal_cstr_len(Object* string, const char* cstr, uword len) {
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

Object* print(Object* obj) {
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
    fprintf(stderr, "unknown tag: %u\n", obj_tag(as_heap_object(obj)));
    abort();
  }
  return obj;
}

Object* println(Object* obj) {
  print(obj);
  putchar('\n');
  return obj;
}

#ifndef MEMORY_SIZE
#define MEMORY_SIZE 4096
#endif

// Put something in the const heap so that __start_const_heap and
// __stop_const_heap are defined by the linker.
#define CONST_HEAP const __attribute__((section("const_heap")))
CONST_HEAP
    __attribute__((used)) struct heap_string private_unused_const_heap = {
        .HEAD.tag = TAG_STRING, .size = 11, .data = "hello world"};
