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
