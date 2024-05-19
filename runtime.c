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

// To implement by the user:
size_t heap_object_size(struct gc_obj *obj);
size_t trace_heap_object(struct gc_obj *obj, struct gc_heap *heap,
                         void (*visit)(struct gc_obj **field,
                                       struct gc_heap *heap));
void trace_roots(struct gc_heap *heap,
                 void (*visit)(struct gc_obj **field,
                               struct gc_heap *heap));

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

void visit_field(struct gc_obj **field, struct gc_heap *heap) {
  struct gc_obj *from = *field;
  struct gc_obj *to =
    is_forwarded(from) ? forwarded(from) : copy(heap, from);
  *field = to;
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

static NEVER_INLINE ALLOCATOR struct gc_obj* allocate_slow_path(
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
  return (struct gc_obj*)addr;
}

static inline ALLOCATOR struct gc_obj* allocate(struct gc_heap *heap, size_t size) {
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  if (UNLIKELY(heap->limit < new_hp)) {
    return allocate_slow_path(heap, size);
  }
  heap->hp = new_hp;
  return (struct gc_obj*)addr;
}

// Application

#define FOREACH_TAG(TAG) \
  TAG(TAG_NUM) \
  TAG(TAG_LIST) \
  TAG(TAG_CLOSURE)
enum {
// All odd becase of the NOT_FORWARDED_BIT
#define ENUM_TAG(TAG) TAG = __COUNTER__ * 2 + 1,
  FOREACH_TAG(ENUM_TAG)
#undef ENUM_TAG
};

struct num {
  struct gc_obj HEAD;
  word value;
};

struct list {
  struct gc_obj HEAD;
  size_t size;
  struct gc_obj* items[];
};


typedef struct gc_obj* (*ClosureFn)(struct gc_obj*, struct gc_obj*);

// TODO(max): Figure out if there is a way to do a PyObject_HEAD version of
// this where each closure actually has its own struct with named members
struct closure {
  struct gc_obj HEAD;
  ClosureFn fn;
  size_t size;
  struct gc_obj* env[];
};

size_t variable_size(size_t base, size_t count) {
  return base + count * kPointerSize;
}

size_t heap_object_size(struct gc_obj *obj) {
  switch(obj->tag) {
  case TAG_NUM:
    return sizeof(struct num);
  case TAG_LIST:
    return variable_size(sizeof(struct list), ((struct list*)obj)->size);
  case TAG_CLOSURE:
    return variable_size(sizeof(struct closure), ((struct closure*)obj)->size);
  default:
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
}

size_t trace_heap_object(struct gc_obj *obj, struct gc_heap *heap,
                         void (*visit)(struct gc_obj **field,
                                       struct gc_heap *heap)) {
  switch(obj->tag) {
  case TAG_NUM:
    break;
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
  default:
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
  return heap_object_size(obj);
}

struct gc_obj* mknum(struct gc_heap *heap, int value) {
  struct num *obj = (struct num*)allocate(heap, sizeof *obj);
  obj->HEAD.tag = TAG_NUM;
  obj->value = value;
  return (struct gc_obj*)obj;
}

bool is_num(struct gc_obj* obj) {
  return obj->tag == TAG_NUM;
}

word num_value(struct gc_obj* obj) {
  assert(is_num(obj));
  return ((struct num*)obj)->value;
}

struct gc_obj* mklist(struct gc_heap *heap, size_t size) {
  assert(size >= 0);
  // TODO(max): Return canonical empty list
  struct list *obj = (struct list*)allocate(heap, variable_size(sizeof *obj, size));
  obj->HEAD.tag = TAG_LIST;
  obj->size = size;
  // Assumes the items will be filled in immediately after calling mklist so
  // they are not initialized
  return (struct gc_obj*)obj;
}

bool is_list(struct gc_obj* obj) {
  return obj->tag == TAG_LIST;
}

size_t list_size(struct gc_obj* obj) {
  assert(is_list(obj));
  return ((struct list*)obj)->size;
}

void list_set(struct gc_obj *list, size_t i, struct gc_obj *item) {
  assert(list->tag == TAG_LIST);
  struct list *l = (struct list*)list;
  assert(i < l->size);
  l->items[i] = item;
}

struct gc_obj* mkclosure(struct gc_heap* heap, ClosureFn fn, size_t size) {
  struct closure *obj = (struct closure*)allocate(heap, variable_size(sizeof *obj, size));
  obj->HEAD.tag = TAG_CLOSURE;
  obj->fn = fn;
  obj->size = size;
  // Assumes the items will be filled in immediately after calling mklist so
  // they are not initialized
  return (struct gc_obj*)obj;
}

bool is_closure(struct gc_obj* obj) {
  return obj->tag == TAG_CLOSURE;
}

void closure_set(struct gc_obj *closure, size_t i, struct gc_obj *item) {
  assert(closure->tag == TAG_CLOSURE);
  struct closure *c = (struct closure*)closure;
  assert(i < c->size);
  c->env[i] = item;
}

struct gc_obj* closure_get(struct gc_obj *closure, size_t i) {
  assert(closure->tag == TAG_CLOSURE);
  struct closure *c = (struct closure*)closure;
  assert(i < c->size);
  return c->env[i];
}

struct handles {
  // TODO(max): Figure out how to make this a flat linked list with whole
  // chunks popped off at function return
  struct gc_obj** stack[20];
  size_t stack_pointer;
  struct handles* next;
};

static struct handles* handles = NULL;

void pop_handles(void* local_handles) {
  (void)local_handles;
  handles = handles->next;
}

#define HANDLES() struct handles local_handles __attribute__((__cleanup__(pop_handles))) = { .next = handles }; handles = &local_handles
#define GC_PROTECT(x) local_handles.stack[local_handles.stack_pointer++] = (struct gc_obj**)(&x)
#define END_HANDLES() handles = local_handles.next
#define GC_HANDLE(type, name, val) type name = val; GC_PROTECT(name)

void trace_roots(struct gc_heap *heap,
                 void (*visit)(struct gc_obj **field,
                               struct gc_heap *heap)) {
  for (struct handles *h = handles; h; h = h->next) {
    for (size_t i = 0; i < h->stack_pointer; i++) {
      visit(h->stack[i], heap);
    }
  }
}

static struct gc_heap *heap = NULL;

struct gc_obj* num_add(struct gc_obj *a, struct gc_obj *b) {
  assert(is_num(a));
  assert(is_num(b));
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)+num_value(b));
}

struct gc_obj* num_sub(struct gc_obj *a, struct gc_obj *b) {
  assert(is_num(a));
  assert(is_num(b));
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)-num_value(b));
}

struct gc_obj* num_mul(struct gc_obj *a, struct gc_obj *b) {
  assert(is_num(a));
  assert(is_num(b));
  // NB: doesn't use pointers after allocating
  return mknum(heap, num_value(a)*num_value(b));
}

struct gc_obj* list_append(struct gc_obj *list_obj, struct gc_obj *item) {
  assert(is_list(list_obj));
  struct list *list = (struct list*)list_obj;
  HANDLES();
  GC_PROTECT(list);
  GC_PROTECT(item);
  struct gc_obj *result = mklist(heap, list->size + 1);
  for (size_t i = 0; i < list->size; i++) {
    list_set(result, i, list->items[i]);
  }
  list_set(result, list->size, item);
  return result;
}

struct gc_obj* print(struct gc_obj *obj) {
  if (obj->tag == TAG_NUM) {
    printf("%ld", num_value(obj));
  } else if (obj->tag == TAG_LIST) {
    struct list *list = (struct list*)obj;
    printf("[");
    for (size_t i = 0; i < list->size; i++) {
      print(list->items[i]);
      if (i + 1 < list->size) {
        printf(", ");
      }
    }
    printf("]");
  } else {
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
  return obj;
}

struct gc_obj* println(struct gc_obj *obj) {
  print(obj);
  printf("\n");
  return obj;
}
