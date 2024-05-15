#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

struct gc_obj {
  union {
    uintptr_t tag;
    struct gc_obj *forwarded; // for GC
  };
  uintptr_t payload[0];
};

static const uintptr_t NOT_FORWARDED_BIT = 1;
int is_forwarded(struct gc_obj *obj) {
  return (obj->tag & NOT_FORWARDED_BIT) == 0;
}
void* forwarded(struct gc_obj *obj) { 
  return obj->forwarded;
}
void forward(struct gc_obj *from, struct gc_obj *to) {
  fprintf(stderr, "forward: %p -> %p\n", from, to);
  from->forwarded = to;
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
}

static inline struct gc_obj* allocate(struct gc_heap *heap, size_t size) {
retry:
  uintptr_t addr = heap->hp;
  uintptr_t new_hp = align_size(addr + size);
  if (heap->limit < new_hp) {
    collect(heap);
    if (heap->limit - heap->hp < size) {
      fprintf(stderr, "out of memory\n");
      abort();
    }
    goto retry;
  }
  heap->hp = new_hp;
  return (struct gc_obj*)addr;
}

// Application

// All odd becase of the NOT_FORWARDED_BIT
enum {
  TAG_CONS = 1,
  TAG_NUM = 3,
};

struct num {
  struct gc_obj HEAD;
  int value;
};

struct cons {
  struct gc_obj HEAD;
  struct gc_obj *car;
  struct gc_obj *cdr;
};

size_t heap_object_size(struct gc_obj *obj) {
  switch(obj->tag) {
  case TAG_NUM:
    return sizeof(struct num);
  case TAG_CONS:
    return sizeof(struct cons);
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
  case TAG_CONS:
    visit(&((struct cons*)obj)->car, heap);
    visit(&((struct cons*)obj)->cdr, heap);
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

struct gc_obj* mkcons(struct gc_heap *heap, struct gc_obj *car, struct gc_obj *cdr) {
  struct cons *obj = (struct cons*)allocate(heap, sizeof *obj);
  obj->HEAD.tag = TAG_CONS;
  obj->car = car;
  obj->cdr = cdr;
  return (struct gc_obj*)obj;
}

struct handles {
  struct gc_obj** stack[10];
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
  struct num *num_a = (struct num*)a;
  struct num *num_b = (struct num*)b;
  return mknum(heap, num_a->value + num_b->value);
}

struct gc_obj* num_mul(struct gc_obj *a, struct gc_obj *b) {
  struct num *num_a = (struct num*)a;
  struct num *num_b = (struct num*)b;
  return mknum(heap, num_a->value * num_b->value);
}

struct gc_obj* print(struct gc_obj *obj) {
  if (obj->tag == TAG_NUM) {
    struct num *num = (struct num*)obj;
    fprintf(stdout, "%d\n", num->value);
  } else {
    fprintf(stderr, "unknown tag: %lu\n", obj->tag);
    abort();
  }
  return obj;
}
