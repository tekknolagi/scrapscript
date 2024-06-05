int main() {
  heap = make_heap(MEMORY_SIZE);
  HANDLES();
  GC_HANDLE(struct object*, result, scrap_main());
  println(result);
  destroy_heap(heap);
  return 0;
}
