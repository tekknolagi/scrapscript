#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

const int kPointerSize = sizeof(void*);
typedef intptr_t word;
typedef uintptr_t uword;

struct gc_obj {
  uintptr_t tag; // low bit is 0 if forwarding ptr
  uintptr_t payload[0];
};

// The low bit of the pointer is 1 if it's a heap object and 0 if it's an
// immediate integer
struct object {};

static const uword kImmediateMask = (uword)1ULL;
static const uword kImmediateBits = 1;
bool is_immediate(struct object* obj) { return (((uword)obj) & kImmediateMask) == 0; }
uword as_immediate(struct object* obj) {
  assert(is_immediate(obj));
  return ((uword)obj) >> kImmediateBits;
}
struct gc_obj* as_heap_object(struct object* obj) {
  assert(!is_immediate(obj));
  return (struct gc_obj*)((uword)obj & ~kImmediateMask);
}

static const uintptr_t NOT_FORWARDED_BIT = 1;
int is_forwarded(struct gc_obj *obj) {
  return (obj->tag & NOT_FORWARDED_BIT) == 0;
}
struct gc_obj* forwarded(struct gc_obj *obj) {
  assert(is_forwarded(obj));
  return (struct gc_obj*)obj->tag;
}
void forward(struct gc_obj *from, struct gc_obj *to) {
  assert(!is_forwarded(from));
  assert((((uintptr_t)to) & NOT_FORWARDED_BIT) == 0);
  from->tag = (uintptr_t)to;
}

struct gc_heap;

typedef void (*VisitFn)(struct object **, struct gc_heap *);

// To implement by the user:
size_t heap_object_size(struct gc_obj *obj);
size_t trace_heap_object(struct gc_obj *obj, struct gc_heap *heap, VisitFn visit);
void trace_roots(struct gc_heap *heap, VisitFn visit);

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

static struct gc_heap* make_heap(size_t size) {
  size = align(size, getpagesize());
  struct gc_heap *heap = malloc(sizeof(struct gc_heap));
  void *mem = mmap(NULL, size, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  heap->to_space = heap->hp = (uintptr_t) mem;
  heap->from_space = heap->limit = heap->hp + size / 2;
  heap->size = size;
  return heap;
}

void destroy_heap(struct gc_heap *heap) {
  munmap((void*)heap->to_space, heap->size);
  free(heap);
}

struct gc_obj* copy(struct gc_heap *heap, struct gc_obj *obj) {
  size_t size = heap_object_size(obj);
  struct gc_obj *new_obj = (struct gc_obj*)heap->hp;
  memcpy(new_obj, obj, size);
  forward(obj, new_obj);
  heap->hp += align_size(size);
  return new_obj;
}

void flip(struct gc_heap *heap) {
  heap->hp = heap->from_space;
  heap->from_space = heap->to_space;
  heap->to_space = heap->hp;
  heap->limit = heap->hp + heap->size / 2;
}

struct object* heap_tag(uintptr_t addr) {
  return (struct object*)(addr | (uword)1ULL);
}

void visit_field(struct object **pointer, struct gc_heap *heap) {
  if (is_immediate(*pointer)) {
    return;
  }
  struct gc_obj *from = as_heap_object(*pointer);
  struct gc_obj *to = is_forwarded(from) ? forwarded(from) : copy(heap, from);
  *pointer = heap_tag((uintptr_t)to);
}

void collect(struct gc_heap *heap) {
  flip(heap);
  uintptr_t scan = heap->hp;
  trace_roots(heap, visit_field);
  while(scan < heap->hp) {
    struct gc_obj *obj = (struct gc_obj*)scan;
    scan += align_size(trace_heap_object(obj, heap, visit_field));
  }
#ifndef NDEBUG
  // Zero out the rest of the heap for debugging
  memset((void*)scan, 0, heap->limit - scan);
#endif
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ALLOCATOR __attribute__((__malloc__))
#define NEVER_INLINE __attribute__((noinline))

static NEVER_INLINE ALLOCATOR struct object* allocate_slow_path(
    struct gc_heap* heap,
    size_t size
) {
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

static inline ALLOCATOR struct object* allocate(struct gc_heap *heap, size_t size) {
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  if (UNLIKELY(heap->limit < new_hp)) {
    return allocate_slow_path(heap, size);
  }
  heap->hp = new_hp;
  return heap_tag(addr);
}

// Application

#define FOREACH_TAG(TAG) \
  TAG(TAG_LIST) \
  TAG(TAG_CLOSURE) \
  TAG(TAG_RECORD)

enum {
// All odd becase of the NOT_FORWARDED_BIT
#define ENUM_TAG(TAG) TAG = __COUNTER__ * 2 + 1,
  FOREACH_TAG(ENUM_TAG)
#undef ENUM_TAG
};

struct list {
  struct gc_obj HEAD;
  size_t size;
  struct object* items[];
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

size_t variable_size(size_t base, size_t count) {
  return base + count * kPointerSize;
}

size_t record_size(size_t count) {
  return sizeof(struct record) + count * sizeof(struct record_field);
}

size_t heap_object_size(struct gc_obj *obj) {
  switch(obj->tag) {
  case TAG_LIST:
    return variable_size(sizeof(struct list), ((struct list*)obj)->size);
  case TAG_CLOSURE:
    return variable_size(sizeof(struct closure), ((struct closure*)obj)->size);
  case TAG_RECORD:
    return record_size(((struct record*)obj)->size);
  default:
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
}

size_t trace_heap_object(struct gc_obj *obj, struct gc_heap *heap, VisitFn visit) {
  switch(obj->tag) {
  case TAG_LIST:
    for (size_t i = 0; i < ((struct list*)obj)->size; i++) {
      visit(&((struct list*)obj)->items[i], heap);
    }
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
  default:
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
  return heap_object_size(obj);
}

struct object* mknum(struct gc_heap *heap, int value) {
  (void)heap;
  // TODO(max): Check if it fits in 63 bits
  return (struct object*)(((word)value << kImmediateBits));
}

bool is_num(struct object* obj) {
  return is_immediate(obj);
}

word num_value(struct object* obj) {
  assert(is_num(obj));
  return ((word)obj) >> 1;  // sign extend
}

bool is_list(struct object* obj) {
  if (is_immediate(obj)) {
    return false;
  }
  return as_heap_object(obj)->tag == TAG_LIST;
}

struct list* as_list(struct object* obj) {
  assert(is_list(obj));
  return (struct list*)as_heap_object(obj);
}
struct object* mklist(struct gc_heap *heap, size_t size) {
  assert(size >= 0);
  // TODO(max): Return canonical empty list
  struct object* result = allocate(heap, variable_size(sizeof(struct list), size));
  as_heap_object(result)->tag = TAG_LIST;
  as_list(result)->size = size;
  // Assumes the items will be filled in immediately after calling mklist so
  // they are not initialized
  return result;
}

size_t list_size(struct object* obj) {
  return as_list(obj)->size;
}

struct object* list_get(struct object *list, size_t i) {
  assert(is_list(list));
  assert(i < list_size(list));
  return as_list(list)->items[i];
}

void list_set(struct object *list, size_t i, struct object *item) {
  assert(is_list(list));
  assert(i < list_size(list));
  as_list(list)->items[i] = item;
}

bool is_closure(struct object* obj) {
  if (is_immediate(obj)) {
    return false;
  }
  return as_heap_object(obj)->tag == TAG_CLOSURE;
}

struct closure* as_closure(struct object* obj) {
  assert(is_closure(obj));
  return (struct closure*)as_heap_object(obj);
}

struct object* mkclosure(struct gc_heap* heap, ClosureFn fn, size_t size) {
  struct object *result = allocate(heap, variable_size(sizeof(struct closure), size));
  as_heap_object(result)->tag = TAG_CLOSURE;
  as_closure(result)->fn = fn;
  as_closure(result)->size = size;
  // Assumes the items will be filled in immediately after calling mklist so
  // they are not initialized
  return result;
}

ClosureFn closure_fn(struct object* obj) {
  return as_closure(obj)->fn;
}

void closure_set(struct object *closure, size_t i, struct object *item) {
  struct closure *c = as_closure(closure);
  assert(i < c->size);
  c->env[i] = item;
}

struct object* closure_get(struct object *closure, size_t i) {
  struct closure *c = as_closure(closure);
  assert(i < c->size);
  return c->env[i];
}

struct object* closure_call(struct object *closure, struct object *arg) {
  ClosureFn fn = closure_fn(closure);
  return fn(closure, arg);
}

bool is_record(struct object* obj) {
  if (is_immediate(obj)) {
    return false;
  }
  return as_heap_object(obj)->tag == TAG_RECORD;
}

struct record* as_record(struct object* obj) {
  assert(is_record(obj));
  return (struct record*)as_heap_object(obj);
}

struct object* mkrecord(struct gc_heap* heap, size_t size) {
  // size is the number of fields, each of which has an index and a value
  // (object)
  struct object *result = allocate(heap, record_size(size));
  as_heap_object(result)->tag = TAG_RECORD;
  as_record(result)->size = size;
  // Assumes the items will be filled in immediately after calling mkrecord so
  // they are not initialized
  return result;
}

void record_set(struct object *record, size_t index, size_t key, struct object *value) {
  struct record *r = as_record(record);
  assert(index < r->size);
  r->fields[index].key = key;
  r->fields[index].value = value;
}

struct object* record_get(struct object *record, size_t key) {
  struct record *r = as_record(record);
  struct record_field *fields = r->fields;
  for (size_t i = 0; i < r->size; i++) {
    struct record_field field = fields[i];
    if (field.key == key) {
      return field.value;
    }
  }
  return NULL;
}

#define MAX_HANDLES 20

struct handles {
  // TODO(max): Figure out how to make this a flat linked list with whole
  // chunks popped off at function return
  struct object** stack[MAX_HANDLES];
  size_t stack_pointer;
  struct handles* next;
};

static struct handles* handles = NULL;

void pop_handles(void* local_handles) {
  (void)local_handles;
  handles = handles->next;
}

#define HANDLES() struct handles local_handles __attribute__((__cleanup__(pop_handles))) = { .next = handles }; handles = &local_handles
#define GC_PROTECT(x) assert(local_handles.stack_pointer < MAX_HANDLES); local_handles.stack[local_handles.stack_pointer++] = (struct object**)(&x)
#define END_HANDLES() handles = local_handles.next
#define GC_HANDLE(type, name, val) type name = val; GC_PROTECT(name)

void trace_roots(struct gc_heap *heap, VisitFn visit) {
  for (struct handles *h = handles; h; h = h->next) {
    for (size_t i = 0; i < h->stack_pointer; i++) {
      visit(h->stack[i], heap);
    }
  }
}

static struct gc_heap *heap = NULL;

struct object* num_add(struct object *a, struct object *b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)+num_value(b));
}

struct object* num_sub(struct object *a, struct object *b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)-num_value(b));
}

struct object* num_mul(struct object *a, struct object *b) {
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)*num_value(b));
}

struct object* list_cons(struct object* item, struct object* list) {
  HANDLES();
  GC_PROTECT(item);
  GC_PROTECT(list);
  size_t size = list_size(list);
  GC_HANDLE(struct object*, result, mklist(heap, size + 1));
  list_set(result, 0, item);
  for (size_t i = 0; i < size; i++) {
    list_set(result, i + 1, list_get(list, i));
  }
  return result;
}

struct object* list_rest(struct object* list) {
  HANDLES();
  assert(list_size(list) > 0);
  size_t new_size = list_size(list) - 1;
  GC_HANDLE(struct object*, result, mklist(heap, new_size));
  for (size_t i = 0; i < new_size; i++) {
    list_set(result, i, list_get(list, i + 1));
  }
  return result;
}

struct object* list_append(struct object *list, struct object *item) {
  HANDLES();
  GC_PROTECT(list);
  GC_PROTECT(item);
  size_t size = list_size(list);
  GC_HANDLE(struct object *, result, mklist(heap, size + 1));
  for (size_t i = 0; i < size; i++) {
    list_set(result, i, list_get(list, i));
  }
  list_set(result, size, item);
  return result;
}

const char* record_keys[];

struct object *print(struct object *obj) {
  if (is_num(obj)) {
    printf("%ld", num_value(obj));
  } else if (is_list(obj)) {
    size_t size = list_size(obj);
    putchar('[');
    for (size_t i = 0; i < size; i++) {
      print(list_get(obj, i));
      if (i + 1 < size) {
        fputs(", ", stdout);
      }
    }
    putchar(']');
  } else if (is_record(obj)) {
    struct record *record = as_record(obj);
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
  } else {
    assert(!is_immediate(obj));
    fprintf(stderr, "unknown tag: %lu\n", as_heap_object(obj)->tag);
    abort();
  }
  return obj;
}

struct object *println(struct object *obj) {
  print(obj);
  putchar('\n');
  return obj;
}
