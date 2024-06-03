#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#define WBY_IMPLEMENTATION
#include "web.h"

// $ CFLAGS="-I examples/11_platforms/web/" ./compiler.py --compile --platform examples/11_platforms/web/web.c  examples/11_platforms/web/handler.scrap
// or
// $ ./compiler.py --platform examples/11_platforms/web/web.c  examples/11_platforms/web/handler.scrap
// $ cc -I examples/11_platforms/web/ output.c

static int
dispatch(struct wby_con *connection, void *userdata)
{
    HANDLES();
    GC_HANDLE(struct object*, handler, (struct object*)userdata);
    GC_HANDLE(struct object*, url, mkstring(heap, connection->request.uri, strlen(connection->request.uri)));
    GC_HANDLE(struct object*, response, closure_call(handler, url));
    assert(is_record(response));
    GC_HANDLE(struct object*, code, record_get(response, Record_code));
    assert(is_num(code));
    GC_HANDLE(struct object*, body, record_get(response, Record_body));
    assert(is_string(body));

    wby_response_begin(connection, num_value(code), string_length(body), NULL, 0);
    // TODO(max): Copy into buffer or strdup
    wby_write(connection, as_heap_string(body)->data, string_length(body));
    wby_response_end(connection);
    return num_value(code) == 200;
}

int main(int argc, const char * argv[])
{
    /* boot scrapscript */
    heap = make_heap(kMemorySize);
    HANDLES();
    GC_HANDLE(struct object*, handler, scrap_main());
    assert(is_closure(handler));

    /* setup config */
    struct wby_config config;
    memset(&config, 0, sizeof(config));
    config.address = "0.0.0.0";
    config.port = 8000;
    config.connection_max = 8;
    config.request_buffer_size = 2048;
    config.io_buffer_size = 8192;
    config.dispatch = dispatch;
    config.userdata = handler;

    GC_PROTECT(config.userdata);

    /* compute and allocate needed memory and start server */
    struct wby_server server;
    size_t needed_memory;
    wby_init(&server, &config, &needed_memory);
    void *memory = calloc(needed_memory, 1);
    printf("serving at http://%s:%d\n", config.address, config.port);
    wby_start(&server, memory);
    while (1) {
        wby_update(&server);
    }
    wby_stop(&server);
    free(memory);
}

