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
int main() {
  heap = make_heap(MEMORY_SIZE);
  HANDLES();
  GC_HANDLE(struct object*, result, scrap_main());
  println(result);
  destroy_heap(heap);
  return 0;
}
