int main() {
#ifdef STATIC_HEAP
  char memory[MEMORY_SIZE] = {0};
  struct space space = make_space(memory, MEMORY_SIZE);
#else
  struct space space = make_space(MEMORY_SIZE);
#endif
  init_heap(heap, space);
  HANDLES();
  GC_HANDLE(struct object*, result, scrap_main());
  println(result);
  destroy_space(space);
  return 0;
}
