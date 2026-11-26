"""Regression tests for the SPARQL endpoint and request-handling security fixes."""

from fastapi.testclient import TestClient
import pytest

from src.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _strlen(literal: str) -> int:
    from src.app import run_sparql_query

    res = run_sparql_query(f'SELECT (STRLEN("{literal}") AS ?n) WHERE {{}}')
    return int(list(res)[0][0])


def _rows(query: str) -> int:
    from src.app import run_sparql_query

    return len(run_sparql_query(query).bindings)


# These call run_sparql_query() directly (the cached path), so they keep
# testing the cache key even when /sparql later stops using the cache.


def test_cache_does_not_merge_queries_differing_inside_literals():
    """Whitespace inside a string literal is significant; the cache must not ignore it."""
    assert _strlen("a  b") == 4
    assert _strlen("a b") == 3
    # and in the other order
    assert _strlen("c d") == 3
    assert _strlen("c  d") == 4


def test_cache_respects_line_comment_termination():
    """A newline ends a '#' comment; collapsing it to a space would comment out LIMIT."""
    assert _rows("SELECT ?n WHERE { BIND(1 AS ?n) } # note LIMIT 0") == 1
    assert _rows("SELECT ?n WHERE { BIND(1 AS ?n) } # note\nLIMIT 0") == 0
    # and in the other order
    assert _rows("SELECT ?m WHERE { BIND(1 AS ?m) } # c\nLIMIT 0") == 0
    assert _rows("SELECT ?m WHERE { BIND(1 AS ?m) } # c LIMIT 0") == 1


def test_endpoint_returns_correct_answers_for_literal_variants(client):
    for literal, n in [("e  f", 4), ("e f", 3)]:
        q = f'SELECT (STRLEN("{literal}") AS ?n) WHERE {{}}'
        r = client.get("/sparql", params={"query": q})
        assert int(r.json()["results"]["bindings"][0]["n"]["value"]) == n


def test_query_cache_is_bounded_by_total_rows():
    from src.app import BoundedResultCache

    cache = BoundedResultCache(max_entries=10, max_total_rows=100, max_entry_rows=60)
    cache.put("a", "A", 50)
    cache.put("b", "B", 40)
    assert cache.get("a") == "A"  # touch "a" so "b" becomes least recently used
    cache.put("c", "C", 30)  # total would be 120 > 100: evict LRU ("b")
    assert cache.get("b") is None
    assert cache.get("a") == "A" and cache.get("c") == "C"
    assert cache.total_rows == 80
    cache.put("big", "X", 61)  # larger than max_entry_rows: not cached
    assert cache.get("big") is None
    assert cache.total_rows == 80


def test_query_cache_is_bounded_by_entry_count():
    from src.app import BoundedResultCache

    cache = BoundedResultCache(max_entries=3, max_total_rows=10**6, max_entry_rows=10**6)
    for i in range(10):
        cache.put(str(i), i, 1)
    assert len(cache) == 3
    assert [cache.get(str(i)) for i in (7, 8, 9)] == [7, 8, 9]


def test_query_cache_replacement_clear_and_disable():
    from src.app import BoundedResultCache

    cache = BoundedResultCache(max_entries=10, max_total_rows=100, max_entry_rows=100)
    cache.put("a", "A1", 30)
    cache.put("a", "A2", 50)  # replacing an entry must not double-count it
    assert cache.get("a") == "A2" and cache.total_rows == 50 and len(cache) == 1
    cache.put("b", "B", 50)  # exactly at the aggregate limit: kept
    assert cache.total_rows == 100 and len(cache) == 2
    cache.clear()
    assert cache.total_rows == 0 and len(cache) == 0 and cache.get("a") is None

    disabled = BoundedResultCache(max_entries=0, max_total_rows=100, max_entry_rows=100)
    disabled.put("a", "A", 1)
    assert disabled.get("a") is None and disabled.total_rows == 0


def test_query_cache_entry_larger_than_aggregate_limit_is_not_cached():
    from src.app import BoundedResultCache

    cache = BoundedResultCache(max_entries=10, max_total_rows=50, max_entry_rows=100)
    cache.put("small", "S", 10)
    cache.put("big", "B", 60)  # fits the per-entry limit but not the aggregate one
    assert cache.get("big") is None
    assert cache.get("small") == "S"  # and it must not flush the existing entries
    assert cache.total_rows == 10


@pytest.mark.parametrize(
    "name", ["SPARQL_CACHE_MAX_ENTRIES", "SPARQL_CACHE_MAX_TOTAL_ROWS", "SPARQL_CACHE_MAX_ENTRY_ROWS"]
)
def test_negative_cache_settings_are_rejected(name):
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(**{name: -1})


def test_resource_queries_still_use_the_bounded_cache(client):
    """The server's own per-resource CONSTRUCT queries are cached (within the bounds)."""
    from src.app import QUERY_CACHE

    QUERY_CACHE.clear()
    r = client.get("/constants/PlanckConstant", headers={"Accept": "text/turtle"})
    assert r.status_code == 200
    assert len(QUERY_CACHE) == 1


# ---------------------------------------------------------------------------
# Patch 3: isolated execution of client queries
# ---------------------------------------------------------------------------

import errno
import urllib.parse
import json
import os
import signal
import threading
import time

import rdflib

import src.app as app_module
from src.config import settings

HEAVY_QUERY = (
    "SELECT (COUNT(*) AS ?n) WHERE { "
    "?a a <https://w3id.org/codata/model/ConstantValue> . "
    "?b a <https://w3id.org/codata/model/Unit> . "
    "?c a <https://w3id.org/codata/model/Version> . "
    "?d a <https://w3id.org/codata/model/Unit> }"
)
SIMPLE_QUERY = "SELECT ?s WHERE { ?s a <https://w3id.org/codata/model/Unit> } ORDER BY ?s LIMIT 3"


def _no_children_left() -> bool:
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


@pytest.fixture(autouse=True)
def _fresh_admission():
    app_module.SPARQL_ADMISSION.reset()
    yield
    app_module.SPARQL_ADMISSION.reset()
    app_module._UNREAPED.clear()


def _fake_execute(behaviour):
    """Replace the query function run inside the child (fork inherits the patch)."""

    def fake(query, accept):
        return behaviour()

    return fake


def test_public_queries_never_touch_the_cache(client, monkeypatch):
    from src.app import QUERY_CACHE

    QUERY_CACHE.clear()
    for allow_unisolated, isolated in [(False, True), (True, False)]:
        monkeypatch.setattr(settings, "SPARQL_ALLOW_UNISOLATED", allow_unisolated)
        monkeypatch.setattr(app_module, "_sparql_isolation_available", lambda v=isolated: v)
        for i in range(3):
            assert client.get("/sparql", params={"query": f"{SIMPLE_QUERY} # {i}"}).status_code == 200
    assert len(QUERY_CACHE) == 0

    # A planted entry for the same text is not served to a client query.
    g = app_module._require_graph()
    import hashlib

    key = f"{id(g)}:{hashlib.sha256(SIMPLE_QUERY.encode()).hexdigest()}"
    QUERY_CACHE.put(key, "poisoned", 1)
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})
    assert r.status_code == 200 and len(r.json()["results"]["bindings"]) == 3
    QUERY_CACHE.clear()


def test_deadline_kills_a_slow_child(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(lambda: time.sleep(60)))
    t0 = time.monotonic()
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "time limit" in r.json()["detail"]
    assert time.monotonic() - t0 < 3
    assert _no_children_left()


def test_deadline_covers_a_stalled_result_transfer(client, monkeypatch):
    """A child that starts writing and then stalls is still killed at the deadline."""
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 1.0)

    def stall_after_header(write_fd, query, accept, parent_pid):
        os.write(write_fd, b'{"kind": "ok", "media_type": "text/plain", "length": 10}\n12345')
        time.sleep(60)
        os._exit(0)

    monkeypatch.setattr(app_module, "_sparql_child_main", stall_after_header)
    t0 = time.monotonic()
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "time limit" in r.json()["detail"]
    assert time.monotonic() - t0 < 3
    assert _no_children_left()


def test_real_slow_query_is_killed_and_health_stays_responsive(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 2.0)
    health: dict = {}

    def probe():
        time.sleep(0.5)
        t0 = time.monotonic()
        health["status"] = client.get("/health").status_code
        health["seconds"] = time.monotonic() - t0

    t = threading.Thread(target=probe)
    t.start()
    r = client.get("/sparql", params={"query": HEAVY_QUERY})
    t.join(10)
    assert r.status_code == 503
    assert health["status"] == 200 and health["seconds"] < 1.0
    assert _no_children_left()


def test_memory_limit_stops_a_query(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_QUERY_MEMORY_MB", 128)

    def allocate():
        blocks = []
        while True:
            blocks.append(bytearray(64 * 1024 * 1024))

    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(allocate))
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "memory" in r.json()["detail"]
    assert _no_children_left()


def test_memory_limit_stops_a_real_query(client, monkeypatch):
    """ORDER BY over a Cartesian product grew a child to ~940 MB in 10 s before the limit."""
    monkeypatch.setattr(settings, "SPARQL_MAX_QUERY_MEMORY_MB", 150)
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 20.0)
    q = (
        "SELECT ?a ?b ?c WHERE { ?a a <https://w3id.org/codata/model/ConstantValue> . "
        "?b a <https://w3id.org/codata/model/Unit> . ?c a <https://w3id.org/codata/model/Unit> } "
        "ORDER BY ?c ?b ?a LIMIT 1"
    )
    r = client.get("/sparql", params={"query": q})
    assert r.status_code == 503 and "memory" in r.json()["detail"]


def test_child_killed_by_signal_is_reported_as_resource_failure(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "_execute_and_serialize", _fake_execute(lambda: os.kill(os.getpid(), signal.SIGKILL))
    )
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "terminated" in r.json()["detail"]
    assert _no_children_left()


def test_unexpected_child_failures_are_server_errors(client, monkeypatch):
    def boom():
        raise ValueError("secret internal detail")

    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(boom))
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 500 and "secret" not in r.text

    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(lambda: os._exit(3)))
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 500
    assert _no_children_left()


def test_child_closes_inherited_descriptors(client, monkeypatch):
    def list_fds():
        targets = []
        for name in os.listdir("/proc/self/fd"):
            try:
                targets.append(os.readlink(f"/proc/self/fd/{name}"))
            except OSError:
                pass
        return json.dumps(targets).encode(), "application/json"

    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(list_fds))
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 200
    targets = r.json()
    assert not [t for t in targets if t.startswith("socket:") or "eventpoll" in t], targets
    assert len(targets) <= 5  # stdin/stdout/stderr, the result pipe, the /proc listing itself


def test_response_byte_limit_in_child(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_RESPONSE_BYTES", 2048)
    r = client.get(
        "/sparql",
        params={"query": "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 200"},
        headers={"Accept": "application/n-triples"},
    )
    assert r.status_code == 400 and "bytes" in r.json()["detail"]


@pytest.mark.parametrize("size,status", [(4096, 200), (4097, 400), (10_000_000, 400)])
def test_response_byte_limit_enforced_exactly_by_parent(client, monkeypatch, size, status):
    """Even if the child does not check, the parent refuses anything above the limit."""
    monkeypatch.setattr(settings, "SPARQL_MAX_RESPONSE_BYTES", 4096)
    monkeypatch.setattr(
        app_module, "_execute_and_serialize", _fake_execute(lambda: (b"x" * size, "text/plain"))
    )
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == status
    if status == 200:
        assert len(r.content) == size
    else:
        assert "limit" in r.json()["detail"]
    assert _no_children_left()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"kind": "ok", "media_type": "text/plain", "length": 3}\nabcdef',  # more than announced
        b'{"kind": "ok", "media_type": "text/plain", "length": 30}\nabc',  # less than announced
        b"not json\n",
        b"x" * 10000,  # no header line
    ],
)
def test_malformed_child_output_is_a_server_error(client, monkeypatch, payload):
    def lying_child(write_fd, query, accept, parent_pid):
        os.write(write_fd, payload)
        os._exit(0)

    monkeypatch.setattr(app_module, "_sparql_child_main", lying_child)
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 500
    assert _no_children_left()


def test_single_row_with_huge_literal_is_refused(client, monkeypatch):
    """Row limits do not bound size: one row of doubled CONCATs is refused by the byte limit."""
    monkeypatch.setattr(settings, "SPARQL_MAX_RESPONSE_BYTES", 1024 * 1024)
    binds = " ".join(f"BIND(CONCAT(?s{i}, ?s{i}) AS ?s{i + 1})" for i in range(16))
    q = f'SELECT ?s16 WHERE {{ BIND("{"x" * 32}" AS ?s0) {binds} }}'  # 32 * 2**16 = 2 MiB
    r = client.get("/sparql", params={"query": q})
    assert r.status_code == 400 and "bytes" in r.json()["detail"]


def test_oversized_result_is_refused(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_RESULT_ROWS", 10)
    r = client.get("/sparql", params={"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 11"})
    assert r.status_code == 400 and "limit of 10" in r.json()["detail"]
    assert client.get("/sparql", params={"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}).status_code == 200


def _slow(seconds):
    def slow():
        time.sleep(seconds)
        return b'{"boolean": true}', "application/sparql-results+json"

    return slow


def test_busy_endpoint_rejects_immediately_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_CONCURRENT_QUERIES", 1)
    monkeypatch.setattr(settings, "SPARQL_MAX_WAITING_QUERIES", 0)
    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(_slow(2)))
    first: dict = {}
    t = threading.Thread(target=lambda: first.update(r=client.get("/sparql", params={"query": "ASK {}"})))
    t.start()
    time.sleep(0.5)
    t0 = time.monotonic()
    r = client.get("/sparql", params={"query": "ASK {}"})
    elapsed = time.monotonic() - t0
    t.join(10)
    assert r.status_code == 503 and r.headers.get("retry-after") == "5"
    assert elapsed < 0.5
    assert first["r"].status_code == 200


def test_waiting_queue_is_bounded_in_size_and_time(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_CONCURRENT_QUERIES", 1)
    monkeypatch.setattr(settings, "SPARQL_MAX_WAITING_QUERIES", 1)
    monkeypatch.setattr(settings, "SPARQL_MAX_WAIT_SECONDS", 0.5)
    monkeypatch.setattr(app_module, "_execute_and_serialize", _fake_execute(_slow(2)))
    results: dict = {}

    def call(name):
        results[name] = client.get("/sparql", params={"query": "ASK {}"})

    threads = [threading.Thread(target=call, args=(n,)) for n in ("running",)]
    threads[0].start()
    time.sleep(0.3)
    waiter = threading.Thread(target=call, args=("waiting",))
    waiter.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    call("overflow")  # queue already full
    assert time.monotonic() - t0 < 0.3
    for t in threads + [waiter]:
        t.join(10)
    assert results["running"].status_code == 200
    assert results["waiting"].status_code == 503  # waited 0.5 s, then gave up
    assert results["overflow"].status_code == 503


def test_fails_closed_without_isolation(client, monkeypatch):
    monkeypatch.setattr(app_module, "_sparql_isolation_available", lambda: False)
    monkeypatch.setattr(settings, "SPARQL_ALLOW_UNISOLATED", False)
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})
    assert r.status_code == 503 and "disabled" in r.json()["detail"]
    monkeypatch.setattr(settings, "SPARQL_ALLOW_UNISOLATED", True)
    assert client.get("/sparql", params={"query": SIMPLE_QUERY}).status_code == 200


def test_fork_failure_is_503_and_leaks_nothing(client, monkeypatch):
    def no_fork():
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    before = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(app_module.os, "fork", no_fork)
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "query process" in r.json()["detail"]
    assert len(os.listdir("/proc/self/fd")) == before


EQUIVALENCE_CASES = [
    (
        "SELECT ?c ?v ?u WHERE { ?c <https://w3id.org/codata/model/hasValue> ?cv . "
        "?cv <https://w3id.org/codata/model/value> ?v . "
        "OPTIONAL { ?cv <https://w3id.org/codata/model/uncertainty> ?u } } ORDER BY ?c ?v LIMIT 500",
        "application/sparql-results+json",
    ),
    (SIMPLE_QUERY, "application/sparql-results+xml"),
    ("ASK { ?s a <https://w3id.org/codata/model/Unit> }", "application/sparql-results+json"),
    (
        "CONSTRUCT { ?cv ?p ?o } WHERE { <https://w3id.org/codata/constants/PlanckConstant> "
        "<https://w3id.org/codata/model/hasValue> ?cv . ?cv ?p ?o }",
        "application/n-triples",
    ),
    ("DESCRIBE <https://w3id.org/codata/constants/ElementaryCharge>", "text/turtle"),
    (
        "CONSTRUCT { ?cv ?p ?o } WHERE { <https://w3id.org/codata/constants/ReducedPlanckConstant> "
        "<https://w3id.org/codata/model/hasValue> ?cv . ?cv ?p ?o }",
        "application/ld+json",
    ),
]


def _canonical(body: bytes, media_type: str):
    if "json" in media_type and "ld+json" not in media_type:
        return json.loads(body)
    if media_type.startswith("application/sparql-results+xml"):
        return body
    fmt = {"application/n-triples": "nt", "text/turtle": "turtle", "application/ld+json": "json-ld"}[
        media_type.split(";")[0]
    ]
    g = rdflib.Graph().parse(data=body, format=fmt)
    return sorted(g)


@pytest.mark.parametrize("query,accept", EQUIVALENCE_CASES)
def test_isolated_and_inprocess_results_are_identical(client, monkeypatch, query, accept):
    """Same RDF terms, datatypes and lexical forms (incl. full-precision values) in both modes."""
    isolated = client.get("/sparql", params={"query": query}, headers={"Accept": accept})
    monkeypatch.setattr(app_module, "_sparql_isolation_available", lambda: False)
    monkeypatch.setattr(settings, "SPARQL_ALLOW_UNISOLATED", True)
    inprocess = client.get("/sparql", params={"query": query}, headers={"Accept": accept})
    assert isolated.status_code == inprocess.status_code == 200
    assert isolated.headers["content-type"] == inprocess.headers["content-type"]
    media = isolated.headers["content-type"]
    assert _canonical(isolated.content, media) == _canonical(inprocess.content, media)
    assert len(isolated.content) > 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("SPARQL_TIMEOUT_SECONDS", 0),
        ("SPARQL_TIMEOUT_SECONDS", -1),
        ("SPARQL_TIMEOUT_SECONDS", float("inf")),
        ("SPARQL_MAX_CONCURRENT_QUERIES", 0),
        ("SPARQL_MAX_WAITING_QUERIES", -1),
        ("SPARQL_MAX_RESULT_ROWS", 0),
        ("SPARQL_MAX_RESPONSE_BYTES", 10),
        ("SPARQL_MAX_QUERY_MEMORY_MB", 1),
    ],
)
def test_invalid_query_limit_settings_are_rejected(name, value):
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(**{name: value})


# --- v3: slot ownership, process supervision, resource-limit set-up, error classes ---


def test_cancelled_request_keeps_its_slot_until_execution_ends(monkeypatch):
    """Cancelling the awaiting request must not free the slot while the query runs."""
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setattr(settings, "SPARQL_MAX_CONCURRENT_QUERIES", 1)
    monkeypatch.setattr(settings, "SPARQL_MAX_WAITING_QUERIES", 0)
    finished = threading.Event()

    def slow_blocking(query, accept):
        time.sleep(1.0)
        finished.set()
        return b"{}", "application/json"

    monkeypatch.setattr(app_module, "_run_public_sparql_blocking", slow_blocking)

    async def scenario():
        first = asyncio.create_task(app_module.execute_public_sparql("ASK {}", ""))
        await asyncio.sleep(0.2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        # The cancelled request's query is still running: no slot may be free.
        with pytest.raises(HTTPException) as busy:
            await app_module.execute_public_sparql("ASK {}", "")
        assert busy.value.status_code == 503
        assert not finished.is_set()
        await asyncio.sleep(1.2)
        assert finished.is_set()
        ok = await app_module.execute_public_sparql("ASK {}", "")
        assert ok.status_code == 200

    asyncio.run(scenario())


def test_child_that_lingers_after_closing_its_pipe_is_bounded(client, monkeypatch):
    """EOF is not exit: a child that closes its pipe and stays alive is killed at the
    deadline, and its result is not used (strict deadline policy)."""
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 1.0)

    def linger(write_fd, query, accept, parent_pid):
        os.write(write_fd, b'{"kind": "ok", "media_type": "text/plain", "length": 2}\nok')
        os.close(write_fd)
        time.sleep(60)
        os._exit(0)

    monkeypatch.setattr(app_module, "_sparql_child_main", linger)
    t0 = time.monotonic()
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert time.monotonic() - t0 < 2.5
    assert r.status_code == 503 and "time limit" in r.json()["detail"]
    assert _no_children_left()


def test_child_that_survives_sigkill_does_not_block_the_worker(monkeypatch):
    """If a child cannot be reaped within the grace period it is recorded and reaped later."""
    pid = os.fork()
    if pid == 0:
        try:
            time.sleep(60)
        finally:
            os._exit(0)
    real_kill = os.kill
    monkeypatch.setattr(app_module.os, "kill", lambda p, s: None)  # simulate a child stuck in the kernel
    monkeypatch.setattr(app_module, "_KILL_GRACE_SECONDS", 0.3)
    t0 = time.monotonic()
    status, killed = app_module._finish_child(pid, deadline=time.monotonic(), kill=True)
    assert status is None and killed and time.monotonic() - t0 < 1.0
    assert pid in app_module._UNREAPED
    monkeypatch.setattr(app_module.os, "kill", real_kill)
    real_kill(pid, signal.SIGKILL)
    time.sleep(0.2)
    app_module._reap_leftovers()
    assert pid not in app_module._UNREAPED
    assert _no_children_left()


def test_query_refused_when_resource_limits_cannot_be_set_up(client, monkeypatch):
    def no_statm(extra_mb):
        raise OSError(2, "No such file or directory: '/proc/self/statm'")

    monkeypatch.setattr(app_module, "_limit_child_memory", no_statm)
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})
    assert r.status_code == 503 and "resource-limited" in r.json()["detail"]
    assert _no_children_left()


def _in_forked_child(fn) -> int:
    """Run fn() in a forked child; return its exit code (0 = fn returned True)."""
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            code = 0 if fn() else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def test_memory_limit_never_relaxes_an_inherited_soft_limit():
    import resource

    def check():
        strict = 8 * 1024**3  # well above current use, below current + 512 MB? use current + 64 MB
        with open("/proc/self/statm") as f:
            current = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        strict = current + 64 * 1024 * 1024
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (strict, hard))
        app_module._limit_child_memory(512)
        soft_after, hard_after = resource.getrlimit(resource.RLIMIT_AS)
        return soft_after == strict and hard_after == hard

    assert _in_forked_child(check) == 0


def test_memory_limit_is_relative_to_inherited_size():
    import resource

    def check():
        with open("/proc/self/statm") as f:
            current = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        app_module._limit_child_memory(256)
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        return current + 255 * 1024**2 < soft <= current + 257 * 1024**2

    assert _in_forked_child(check) == 0


def _process_gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            return any(line.startswith("State:") and "Z" in line for line in f)
    except FileNotFoundError:
        return True


def test_query_child_dies_with_its_parent():
    """A worker killed by its process manager must not leave query processes behind."""
    r, w = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:  # stands in for a worker process
        try:
            os.close(r)
            me = os.getpid()
            child = os.fork()
            if child == 0:
                try:
                    app_module._set_parent_death_signal(me)
                    time.sleep(60)
                finally:
                    os._exit(0)
            os.write(w, str(child).encode())
            time.sleep(60)
        finally:
            os._exit(0)
    os.close(w)
    time.sleep(0.5)
    grandchild = int(os.read(r, 32))
    os.close(r)
    os.kill(supervisor, signal.SIGKILL)
    os.waitpid(supervisor, 0)
    for _ in range(50):
        if _process_gone(grandchild):
            break
        time.sleep(0.05)
    assert _process_gone(grandchild)


@pytest.mark.parametrize("where", ["prepare", "evaluate"])
@pytest.mark.parametrize("isolated", [True, False])
def test_unexpected_engine_exceptions_are_500_without_details(client, monkeypatch, where, isolated):
    def boom(*args, **kwargs):
        raise TypeError("secret internal detail")

    if where == "prepare":
        monkeypatch.setattr(app_module, "prepareQuery", boom)
    else:
        monkeypatch.setattr(rdflib.Graph, "query", boom)
    if not isolated:
        monkeypatch.setattr(app_module, "_sparql_isolation_available", lambda: False)
        monkeypatch.setattr(settings, "SPARQL_ALLOW_UNISOLATED", True)
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})
    assert r.status_code == 500
    assert "secret" not in r.text


@pytest.mark.parametrize(
    "query,fragment",
    [
        ("SELEC * WHERE {}", "SPARQL query error"),
        ("SELECT * WHERE { foo:x ?p ?o }", "prefix"),
        ('SELECT * WHERE { BIND(REGEX("a", "[") AS ?x) }', "SPARQL query error"),
        ('SELECT * WHERE { BIND(STRLANG("a", 5) AS ?x) }', "SPARQL query error"),
        ("SELECT * WHERE { ?s ?p ?o } LIMIT 99999999999999999999999", "SPARQL query error"),
        ("SELECT * WHERE " + "{" * 3000 + "}" * 3000, "nested too deeply"),
    ],
)
def test_mistakes_in_queries_are_400(client, query, fragment):
    r = client.post("/sparql", content=query, headers={"Content-Type": "application/sparql-query"})
    assert r.status_code == 400, r.text
    assert fragment.lower() in r.json()["detail"].lower()


def test_child_never_returns_into_server_code(client, monkeypatch):
    """If the child's main function raises instead of exiting, the child must still exit."""
    monkeypatch.setattr(settings, "SPARQL_TIMEOUT_SECONDS", 5.0)

    def raising_child(write_fd, query, accept, parent_pid):
        raise TypeError("child escaped")

    monkeypatch.setattr(app_module, "_sparql_child_main", raising_child)
    t0 = time.monotonic()
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 500
    assert time.monotonic() - t0 < 2  # an escaped child would hang until the deadline
    assert _no_children_left()


def test_isolated_execution_preserves_every_value_string_exactly(client):
    """Every codata:value / codata:uncertainty literal comes back byte-for-byte, with its datatype."""
    g = app_module._require_graph()
    model = "https://w3id.org/codata/model/"
    expected = {}
    for prop in ("value", "uncertainty"):
        for s, o in g.subject_objects(rdflib.URIRef(model + prop)):
            expected[(str(s), prop)] = (str(o), str(o.datatype) if o.datatype else None)
    assert len(expected) > 2000

    q = (
        f"SELECT ?cv ?p ?v WHERE {{ VALUES ?p {{ <{model}value> <{model}uncertainty> }} ?cv ?p ?v }}"
    )
    r = client.get("/sparql", params={"query": q}, headers={"Accept": "application/sparql-results+json"})
    assert r.status_code == 200
    got = {}
    for b in r.json()["results"]["bindings"]:
        got[(b["cv"]["value"], b["p"]["value"].rsplit("/", 1)[-1])] = (
            b["v"]["value"],
            b["v"].get("datatype"),
        )
    assert got == expected


# --- v4: deadline policy, stuck processes, bounded diagnostics ---


@pytest.mark.parametrize(
    "after_report,status,fragment",
    [
        ("exit3", 500, "server error"),  # reported success, then exited abnormally
        ("sigkill", 500, "server error"),  # reported success, then died from a signal we did not send
    ],
)
def test_result_is_used_only_if_child_exits_cleanly(client, monkeypatch, after_report, status, fragment):
    def child(write_fd, query, accept, parent_pid):
        os.write(write_fd, b'{"kind": "ok", "media_type": "text/plain", "length": 2}\nok')
        os.close(write_fd)
        if after_report == "exit3":
            os._exit(3)
        os.kill(os.getpid(), signal.SIGKILL)

    monkeypatch.setattr(app_module, "_sparql_child_main", child)
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == status and fragment in r.json()["detail"]
    assert _no_children_left()


def test_queries_refused_while_a_process_could_not_be_reaped(client):
    stuck = os.fork()
    if stuck == 0:
        try:
            time.sleep(60)
        finally:
            os._exit(0)
    app_module._UNREAPED.add(stuck)  # as _finish_child does when SIGKILL did not take effect
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})
    assert r.status_code == 503 and "temporarily refused" in r.json()["detail"]
    os.kill(stuck, signal.SIGKILL)
    time.sleep(0.2)
    r = client.get("/sparql", params={"query": SIMPLE_QUERY})  # reaped at the start of this query
    assert r.status_code == 200
    assert not app_module._UNREAPED
    assert _no_children_left()


@pytest.mark.parametrize(
    "prefix",
    ["p" * 100, "p" * 5000, "p" * 50_000, "é" * 3000, "a" + "é中" * 3000],
    ids=["short", "5k", "50k", "escaped-6x", "mixed-escaped"],
)
def test_long_query_error_messages_stay_400(client, prefix):
    """Error messages quote the query; they must fit the IPC header and stay client errors."""
    r = client.post(
        "/sparql",
        content=f"SELECT * WHERE {{ {prefix}:x ?p ?o }}".encode(),
        headers={"Content-Type": "application/sparql-query"},
    )
    assert r.status_code == 400, r.text[:200]
    detail = r.json()["detail"]
    assert "prefix" in detail.lower()
    assert len(detail.encode()) < app_module._HEADER_MAX_BYTES


def test_encoded_header_is_always_bounded():
    for message in ["x" * 10**6, "é" * 10**5, "\U0001f600" * 10**5, '"\\' * 10**5]:
        encoded = app_module._encode_header({"kind": "query_error", "message": message})
        assert len(encoded) < app_module._HEADER_MAX_BYTES
        assert json.loads(encoded)["kind"] == "query_error"


def test_parent_death_signal_is_the_first_setup_step(monkeypatch):
    """The orphan protection is installed before any other post-fork work."""
    calls = []
    monkeypatch.setattr(app_module, "_set_parent_death_signal", lambda pid: calls.append("pdeathsig"))
    monkeypatch.setattr(app_module.gc, "freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(app_module, "_close_inherited_fds", lambda keep: calls.append("fds"))
    monkeypatch.setattr(app_module, "_limit_child_memory", lambda mb: calls.append("rlimit"))
    monkeypatch.setattr(app_module, "_execute_and_serialize", lambda q, a: (b"", "text/plain"))
    monkeypatch.setattr(app_module, "_write_result", lambda fd, h, b=b"": None)
    monkeypatch.setattr(app_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit):
        app_module._sparql_child_main(-1, "ASK {}", "", os.getppid())
    assert calls == ["pdeathsig", "freeze", "fds", "rlimit"]


# --- v5: exact enforcement of the strict exit policy ---


def _reporting_child(write_fd, query, accept, parent_pid):
    os.write(write_fd, b'{"kind": "ok", "media_type": "text/plain", "length": 2}\nok')
    os.close(write_fd)
    os._exit(0)


def test_forced_cleanup_means_timeout_even_with_a_clean_exit_status(client, monkeypatch):
    """The child exits normally just after the deadline, before SIGKILL lands: still a timeout."""
    real_finish = app_module._finish_child

    def late_but_clean(pid, deadline, kill):
        status, _ = real_finish(pid, deadline, kill)  # really reap it (status 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        return status, True  # ...but forced cleanup had been started

    monkeypatch.setattr(app_module, "_sparql_child_main", _reporting_child)
    monkeypatch.setattr(app_module, "_finish_child", late_but_clean)
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 503 and "time limit" in r.json()["detail"]
    assert _no_children_left()


def test_unknown_exit_status_is_not_a_clean_exit(client, monkeypatch):
    """If the child was reaped elsewhere (ChildProcessError), its result is not used."""
    real_waitpid = os.waitpid

    def reaped_elsewhere(pid, options):
        real_waitpid(pid, 0)  # somebody else reaps it...
        raise ChildProcessError(10, "No child processes")  # ...so we cannot

    monkeypatch.setattr(app_module, "_sparql_child_main", _reporting_child)
    monkeypatch.setattr(app_module.os, "waitpid", reaped_elsewhere)
    r = client.get("/sparql", params={"query": "ASK {}"})
    assert r.status_code == 500
    assert _no_children_left()


def test_wait_child_reports_unknown_status_for_a_process_that_is_not_ours():
    assert app_module._wait_child(1, time.monotonic()) == app_module._STATUS_UNKNOWN
# ---------------------------------------------------------------------------
# Patch 4: X-Forwarded-Prefix and redirects
# ---------------------------------------------------------------------------

from urllib.parse import urljoin, urlsplit

ORIGIN = "https://api.example.org"


def _redirect_target(response, request_url: str) -> str:
    """Resolve Location like a browser and return the absolute URL.

    urljoin() alone is not enough: browsers (WHATWG URL) also treat "%2e" as "."
    when removing dot segments, so those are decoded before resolving.
    """
    import re as _re

    assert response.status_code == 307
    location = _re.sub(r"%2e", ".", response.headers["location"], flags=_re.IGNORECASE)
    return urljoin(request_url, location)


@pytest.mark.parametrize(
    "prefix",
    [
        "//evil.example",
        "https://evil.example",
        "/a//b",
        "/a/../b",
        "/a/./b",
        "/a b",
        '/x"><script>',
        "evil",
        "/a\\b",
        "\\\\evil.example",
        "/a%2Fb",
        "/a//",
        "/",
        "",
    ],
)
def test_malformed_forwarded_prefix_is_ignored(client, prefix):
    r = client.get("/constants/", headers={"X-Forwarded-Prefix": prefix}, follow_redirects=False)
    assert r.headers["location"] == "/constants"

    html = client.get("/constants", headers={"X-Forwarded-Prefix": prefix, "Accept": "text/html"})
    assert html.status_code == 200
    assert "evil.example" not in html.text
    assert '"><script>' not in html.text
    assert 'href="//' not in html.text


@pytest.mark.parametrize("prefix", ["/drum/constants", "/drum/constants/", "/drum-api", "/a.b~c_d"])
def test_wellformed_forwarded_prefix_still_works(client, prefix):
    r = client.get("/constants/?q=planck", headers={"X-Forwarded-Prefix": prefix}, follow_redirects=False)
    assert r.headers["location"] == prefix.rstrip("/") + "/constants?q=planck"


def test_forwarded_prefix_allowlist(client, monkeypatch):
    monkeypatch.setattr(settings, "FORWARDED_PREFIX_ALLOWED", ["/drum/constants"])
    ok = client.get("/constants/", headers={"X-Forwarded-Prefix": "/drum/constants"}, follow_redirects=False)
    assert ok.headers["location"] == "/drum/constants/constants"
    other = client.get("/constants/", headers={"X-Forwarded-Prefix": "/other"}, follow_redirects=False)
    assert other.headers["location"] == "/constants"


def test_forwarded_prefix_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(settings, "TRUST_X_FORWARDED_PREFIX", False)
    r = client.get("/constants/", headers={"X-Forwarded-Prefix": "/drum/constants"}, follow_redirects=False)
    assert r.headers["location"] == "/constants"


def test_configured_root_path_is_kept_without_header():
    with TestClient(app, root_path="/drum/constants") as c:
        r = c.get("/constants/", follow_redirects=False)
        assert r.headers["location"] == "/drum/constants/constants"


@pytest.mark.parametrize("value", ["//evil", "/a/../b", "relative", "/a b"])
def test_invalid_allowlist_entries_are_rejected(value):
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(FORWARDED_PREFIX_ALLOWED=[value])


@pytest.mark.parametrize(
    "path",
    [
        "//evil.example/",
        "///evil.example/",
        "//evil.example/x/",
        "/%5Cevil.example/",
        "/%5C%5Cevil.example/",
        "/%5C/evil.example/",
        "/%2F%2Fevil.example/",
        "/constants//",
    ],
)
def test_trailing_slash_redirect_stays_on_origin(client, path):
    # Pass a full URL: a bare "//host/" would be read by the test client as a host.
    r = client.get(f"http://testserver{path}", follow_redirects=False)
    target = urlsplit(_redirect_target(r, f"{ORIGIN}{path}"))
    assert (target.scheme, target.netloc) == ("https", "api.example.org")
    assert not r.headers["location"].startswith("//")
    assert "\\" not in r.headers["location"]


def test_bad_prefix_warning_is_rate_limited(client, caplog):
    import logging

    app_module._warn_bad_prefix._last = float("-inf")
    with caplog.at_level(logging.WARNING, logger="src.app"):
        for _ in range(20):
            client.get("/health", headers={"X-Forwarded-Prefix": "//evil.example"})
    assert len([r for r in caplog.records if "X-Forwarded-Prefix" in r.getMessage()]) == 1


def _raw_asgi_get(path: str, root_path: str = "", host: bytes = b"testserver") -> tuple[int, dict]:
    """Send a GET with an unnormalised path straight to the ASGI app (HTTP clients
    and browsers remove dot segments before sending, a raw client need not)."""
    import asyncio

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": root_path,
        "query_string": b"", "headers": [(b"host", host)],
        "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
    }
    sent: list = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}


@pytest.mark.parametrize("path", ["/a/../", "/../../evil/", "/constants/./", "/./", "/x/../../", "/units/."])
def test_paths_with_dot_segments_are_not_redirected(path):
    for root in ("", "/drum/constants"):
        status, headers = _raw_asgi_get(path, root, host=b"evil.example")
        assert status == 404, (path, root, headers.get("location"))


@pytest.mark.parametrize("path", ["/constants/", "//x/", "/units/abc/"])
def test_no_redirect_uses_the_host_header(path):
    status, headers = _raw_asgi_get(path, host=b"evil.example")
    assert status == 307
    assert "evil.example" not in headers["location"]


@pytest.mark.parametrize("path", ["/constants/", "//x/", "/units/abc/", "/%5Cevil/"])
def test_redirects_stay_inside_the_root_path(path):
    with TestClient(app, root_path="/drum/constants") as c:
        r = c.get(f"http://testserver{path}", follow_redirects=False)
        target = urlsplit(_redirect_target(r, f"{ORIGIN}/drum/constants{path}"))
        assert target.netloc == "api.example.org"
        assert target.path.startswith("/drum/constants/")


def test_slashes_only_path_redirects_to_root_without_loop():
    with TestClient(app, root_path="/drum/constants") as c:
        r = c.get("http://testserver///", follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"] == "/drum/constants/"
        # A proxy that strips the prefix forwards that as "/", which is served directly.
        assert c.get("/", follow_redirects=False).status_code == 200
    r = TestClient(app).get("http://testserver///", follow_redirects=False)
    assert r.headers["location"] == "/"


# --- v4: decoded path vs encoded URL in redirects ---


@pytest.mark.parametrize(
    "raw_decoded_path,expected_location",
    [
        # request path as received   ->  decoded ASGI path (what the middleware sees)
        ("/%2e%2e/", "/drum/constants/%252e%252e"),  # from /%252e%252e/
        ("/%2E%2e/x/", "/drum/constants/%252E%252e/x"),
        ("/a?b/", "/drum/constants/a%3Fb"),  # from /a%3Fb/: "?" must stay path data
        ("/a#b/", "/drum/constants/a%23b"),  # from /a%23b/
        ("/a%25/", "/drum/constants/a%2525"),  # from /a%2525/: no second decoding
        ("/a%3F/", "/drum/constants/a%253F"),  # from /a%253F/
        ("/café/", "/drum/constants/caf%C3%A9"),
        ("/a b/", "/drum/constants/a%20b"),
    ],
)
def test_redirect_reencodes_the_decoded_path(raw_decoded_path, expected_location):
    status, headers = _raw_asgi_get(raw_decoded_path, root_path="/drum/constants")
    assert status == 307
    location = headers["location"]
    assert location == expected_location
    target = urlsplit(
        _redirect_target(type("R", (), {"status_code": 307, "headers": headers})(), f"{ORIGIN}/drum/constants/x/")
    )
    assert target.netloc == "api.example.org" and target.path.startswith("/drum/constants/")
    assert target.query == "" and target.fragment == ""


def test_query_string_is_kept_verbatim_on_redirect():
    with TestClient(app, root_path="/drum/constants") as c:
        r = c.get("/constants/?q=a%20b&x=%2e%2e", follow_redirects=False)
        assert r.headers["location"] == "/drum/constants/constants?q=a%20b&x=%2e%2e"


def test_redirect_when_server_includes_root_path_in_path():
    """uvicorn --root-path puts the prefix into scope["path"] as well; do not double it."""
    status, headers = _raw_asgi_get("/drum/constants/constants/", root_path="/drum/constants")
    assert status == 307 and headers["location"] == "/drum/constants/constants"
# ---------------------------------------------------------------------------
# Patch 5: input limits and validation on /sparql
# ---------------------------------------------------------------------------

FORM = "application/x-www-form-urlencoded"
RAW = "application/sparql-query"
JSON_CT = "application/json"


def _post(client, content_type, content):
    return client.post("/sparql", content=content, headers={"Content-Type": content_type})


def test_body_over_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_REQUEST_BYTES", 2048)
    r = _post(client, RAW, b" " * 4000 + b"ASK {}")
    assert r.status_code == 413


def test_chunked_body_over_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_REQUEST_BYTES", 2048)

    def chunks():
        for _ in range(10):
            yield b" " * 1000

    r = client.post("/sparql", content=chunks(), headers={"Content-Type": RAW})
    assert r.status_code == 413


def test_query_length_limits(client, monkeypatch):
    monkeypatch.setattr(settings, "SPARQL_MAX_QUERY_LENGTH", 200)
    long_query = "ASK {}" + " " * 300
    assert client.get("/sparql", params={"query": long_query}).status_code == 414
    assert _post(client, RAW, long_query.encode()).status_code == 413
    assert _post(client, JSON_CT, json.dumps({"query": long_query})).status_code == 413
    assert client.get("/sparql", params={"query": "ASK {}"}).status_code == 200


@pytest.mark.parametrize(
    "content_type,content",
    [
        (RAW, b"ASK { ?s ?p \xff\xfe }"),
        (FORM, b"query=ASK%20%7B%7D%FF"),
        (FORM, b"query=ASK \xff"),
        (JSON_CT, b'{"query": "ASK {} \xff"}'),
    ],
)
def test_invalid_utf8_is_400(client, content_type, content):
    r = _post(client, content_type, content)
    assert r.status_code == 400


@pytest.mark.parametrize(
    "content_type,content",
    [
        (JSON_CT, '{"query": 5}'),
        (JSON_CT, '{"query": ["ASK {}"]}'),
        (JSON_CT, '{"query": null}'),
        (JSON_CT, "{}"),
        (JSON_CT, '["ASK {}"]'),
        (JSON_CT, '"ASK {}"'),
        (JSON_CT, "not json"),
        (JSON_CT, '{"query": "   "}'),
        (RAW, ""),
        (RAW, " \n\t "),
        (FORM, ""),
        (FORM, "query="),
        (FORM, "query=%20%20"),
        (FORM, "other=1"),
        (FORM, "query=ASK%20%7B%7D&query=ASK%20%7B%7D"),
    ],
)
def test_malformed_query_input_is_400(client, content_type, content):
    r = _post(client, content_type, content)
    assert r.status_code == 400, r.text


def test_whitespace_only_get_query_is_400(client):
    assert client.get("/sparql", params={"query": "   "}).status_code == 400


def test_unsupported_media_type_is_415(client):
    assert _post(client, "text/plain", "ASK {}").status_code == 415


@pytest.mark.parametrize(
    "content_type,content",
    [
        (RAW, SIMPLE_QUERY),
        (f"{RAW}; charset=utf-8", SIMPLE_QUERY),
        (FORM, urllib.parse.urlencode({"query": SIMPLE_QUERY})),
        (JSON_CT, json.dumps({"query": SIMPLE_QUERY})),
    ],
)
def test_post_encodings_give_the_same_result_as_get(client, content_type, content):
    expected = client.get("/sparql", params={"query": SIMPLE_QUERY}).json()
    r = _post(client, content_type, content)
    assert r.status_code == 200
    assert r.json() == expected


@pytest.mark.parametrize("name,value", [("SPARQL_MAX_REQUEST_BYTES", 10), ("SPARQL_MAX_QUERY_LENGTH", 5)])
def test_invalid_input_limit_settings_are_rejected(name, value):
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(**{name: value})


# --- v3: remaining input-validation edge cases ---


def test_deeply_nested_json_is_400(client):
    body = '{"query": ' + "[" * 100_000 + "]" * 100_000 + "}"  # 200 kB, well under the body limit
    assert _post(client, JSON_CT, body).status_code == 400


@pytest.mark.parametrize("escape", ["\\ud800", "\\udfff", "\\ud800x"])
def test_lone_surrogate_in_json_query_is_400(client, escape):
    r = _post(client, JSON_CT, '{"query": "ASK {} # ' + escape + '"}')
    assert r.status_code == 400 and "unicode" in r.json()["detail"].lower()


def test_valid_surrogate_pair_in_json_query_is_accepted(client):
    r = _post(client, JSON_CT, '{"query": "ASK {} # \\ud83d\\ude00"}')  # U+1F600 as an escaped pair
    assert r.status_code == 200


@pytest.mark.parametrize(
    "content",
    [
        "query=ASK%20%7B%7D&query=",  # second, blank query parameter
        "query=&query=ASK%20%7B%7D",
        "query=ASK%zz%7B%7D",  # malformed percent-escape
        "query=ASK%2",
        "query=ASK%",
    ],
)
def test_form_duplicates_and_malformed_escapes_are_400(client, content):
    assert _post(client, FORM, content).status_code == 400


def test_duplicate_query_parameters_are_400_in_every_encoding(client):
    assert client.get("/sparql?query=ASK%20%7B%7D&query=ASK%20%7B%7D").status_code == 400
    assert client.get("/sparql?query=ASK%20%7B%7D&query=").status_code == 400
    assert _post(client, JSON_CT, '{"query": "ASK {}", "query": "ASK {}"}').status_code == 400
    assert _post(client, FORM, "query=ASK%20%7B%7D&query=ASK%20%7B%7D").status_code == 400
    # a nested object may still use the key "query"
    assert _post(client, JSON_CT, '{"query": "ASK {}", "meta": {"query": 1}}').status_code == 200


# --- v4: GET query strings are parsed as strictly as form bodies ---


@pytest.mark.parametrize(
    "qs",
    [
        "query=SELECT%20(STRLEN(%22%FF%22)%20AS%20%3Fn)%20WHERE%20%7B%7D",  # invalid UTF-8 escape
        "query=ASK%20%7B%7D%20%23%C3",  # truncated UTF-8 sequence
        "query=SELECT%20(STR(%22a%zzb%22)%20AS%20%3Fn)%20WHERE%20%7B%7D",  # malformed escape
        "query=ASK%20%7B%7D%2",
        "query=ASK%20%7B%7D&format=json%",
    ],
)
def test_get_query_string_is_parsed_strictly(client, qs):
    r = client.get(f"/sparql?{qs}")
    assert r.status_code == 400, r.text


def test_get_with_valid_utf8_escapes_still_works(client):
    r = client.get("/sparql?query=SELECT%20(STRLEN(%22%C3%A9%F0%9F%98%80%22)%20AS%20%3Fn)%20WHERE%20%7B%7D")
    assert r.status_code == 200
    assert r.json()["results"]["bindings"][0]["n"]["value"] == "2"


def test_get_plus_means_space_as_before(client):
    r = client.get("/sparql?query=SELECT+(STRLEN(%22a+b%22)+AS+%3Fn)+WHERE+%7B%7D")
    assert r.json()["results"]["bindings"][0]["n"]["value"] == "3"
