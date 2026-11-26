# Limits on the public SPARQL endpoint

`/sparql` executes queries supplied by anonymous clients. On Linux, each query
runs in a separate child process forked from the API worker, with the limits
below. All settings can be overridden by environment variables (or `.env`);
invalid values stop the service at startup.

| Setting | Default | Effect |
|---|---|---|
| `SPARQL_TIMEOUT_SECONDS` | 10 | Deadline for evaluation, serialization, transfer of the result and the child's exit. A result is delivered only if the child has also exited normally by the deadline (checked by polling every ~2 ms, so not exact to the millisecond). When it passes, the child is killed (and reaped within a further 2 s grace period); the client gets 503. |
| `SPARQL_MAX_QUERY_MEMORY_MB` | 512 | Address space the child may add to what it inherited from the worker (`RLIMIT_AS`, never above an inherited limit). Exceeding it gives 503. |
| `SPARQL_MAX_CONCURRENT_QUERIES` | 4 | Queries executing at the same time. A slot is held until the query process has finished, even if the request is cancelled or the client disconnects. |
| `SPARQL_MAX_WAITING_QUERIES` | 4 | Queries allowed to wait for a free slot; further requests get 503 with `Retry-After` at once. |
| `SPARQL_MAX_WAIT_SECONDS` | 2 | How long a waiting query may wait before it gets 503. |
| `SPARQL_MAX_RESULT_ROWS` | 100 000 | Rows (SELECT) or triples (CONSTRUCT/DESCRIBE) in a result; more gives 400. The full graph (~60 000 triples) fits. |
| `SPARQL_MAX_RESPONSE_BYTES` | 32 MiB | Size of the serialized result, checked in the child and again, exactly, by the worker while it reads; more gives 400. The largest full-dataset response today is about 17 MB (`SELECT *` as SPARQL XML; N-Triples 9.4 MB, Turtle 2.8 MB). |
| `SPARQL_ALLOW_UNISOLATED` | false | Query isolation needs Linux. Elsewhere `/sparql` answers 503 unless this is set; with it, queries run in-process **without time or memory limits**: local development only. |

Client queries are never cached; the result cache (`SPARQL_CACHE_*`) only holds
the server's own per-resource queries.

## The query process

Before it evaluates anything, the child

- asks the kernel to kill it if its parent dies (`PR_SET_PDEATHSIG`), so a
  worker killed by its process manager cannot leave query processes behind;
- closes every file descriptor it inherited other than standard
  input/output/error and its result pipe (the listening socket and client
  connections would otherwise stay open in it);
- installs its address-space limit.

This is a resource-limited process, not a sandbox: it can still read files and
open network connections with the server's permissions (see "Not covered by the
application" below).

If the parent-death signal or the memory limit cannot be installed, or the
inherited descriptors cannot be listed, the query is refused with 503 rather
than run without the protection. (Closing a descriptor that is already closed
is not an error.)

Error messages from the query engine are passed to the client with 400 and
truncated to fit the internal result header (about 4 kB). Which exceptions
count as mistakes in the query is a heuristic based on the exception types
rdflib 7.6 raises for them (see `_is_client_query_error`); an internal
`ValueError` would also be reported as 400. Re-run the tests when upgrading
rdflib or pyparsing.

## Capacity planning (all limits are per worker)

Each Gunicorn/Uvicorn worker process has its own slots and queue. With `W`
workers and `C = SPARQL_MAX_CONCURRENT_QUERIES`, plan for:

- up to `W × C` query processes at once;
- per query process: up to `SPARQL_MAX_QUERY_MEMORY_MB` of new address space,
  plus copy-on-write copies of inherited pages that it touches (up to the
  worker's own size in the worst case). `RLIMIT_AS` limits address space, not
  physical memory, so a container memory limit is still needed;
- per running query in the worker: up to about twice `SPARQL_MAX_RESPONSE_BYTES`
  while a result is received and handed to the server;
- responses that have been produced but not yet sent to slow clients: these are
  no longer counted against a slot. Proxy response buffering and connection
  limits bound them.

Measure memory under representative concurrent queries and set the container
limit with margin.

## Not covered by the application

These belong to the deployment:

- **Rate limiting** per client at the reverse proxy.
- **Request size**: keep the proxy's body limit (nginx `client_max_body_size`,
  default 1 MB) in place; the application itself does not limit request bodies.
- **Network access**: query processes need no outbound network access.
  `SERVICE` is refused and remote graph loading is disabled in the
  application, but a network policy that blocks egress is the robust guarantee.
- **File access**: the process needs to read only the data directory. A
  read-only filesystem protects integrity but does not stop reading; expose
  only what the service needs (container image, mounts, permissions).
- **Direct access**: the backend port should be reachable only from the proxy.
- **Worker watchdog**: run under Gunicorn with
  `-k uvicorn.workers.UvicornWorker --timeout 60`, so a worker whose event loop
  stops responding for any reason is restarted. Its query processes die with it.

## Known limitations

- A query keeps running until it finishes or reaches the deadline even if the
  client disconnects, and it keeps its slot until then.
- The child is created with `fork()` from a multi-threaded process (Python 3.12+
  warns about this). The child freezes inherited objects, closes inherited
  descriptors and only evaluates the query, and the deadline kills it if it
  hangs. That covers a child stuck on a lock inherited from another thread, but
  not every conceivable fork-related failure. A supervised pool of query workers
  started outside the request-handling threads would avoid the issue, at the cost
  of more code.
- A process that cannot be reaped within the grace period after `SIGKILL` (e.g.
  stuck in the kernel) is logged and reaped later; it does not block the
  worker's threads. Until it is gone, the worker refuses new queries (503), so
  the number of query processes stays within `SPARQL_MAX_CONCURRENT_QUERIES`.
  It is reaped when the next query arrives; without traffic it may remain a
  zombie (no memory, one process-table entry) until then.
