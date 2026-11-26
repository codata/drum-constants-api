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


def test_large_results_are_not_cached(client):
    """A full-graph CONSTRUCT (~60k triples) must not be kept in the cache."""
    from src.app import QUERY_CACHE

    before = QUERY_CACHE.total_rows
    for i in range(3):
        q = f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o }} # {i}"
        r = client.get("/sparql", params={"query": q}, headers={"Accept": "application/n-triples"})
        assert r.status_code == 200
    assert QUERY_CACHE.total_rows == before
