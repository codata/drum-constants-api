# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-11-27

### Security
- **Exact SPARQL Cache Key (`src/app.py`)**: Key `QUERY_CACHE` on the exact query string instead of whitespace-collapsed text, preventing string literals and `#` line comments from colliding across distinct queries.
- **Bounded SPARQL LRU Cache (`src/app.py`, `src/config.py`)**: Replaced the unbounded cache dict with `BoundedResultCache` (LRU with configurable `SPARQL_CACHE_MAX_ENTRIES=512`, `SPARQL_CACHE_MAX_TOTAL_ROWS=100000`, `SPARQL_CACHE_MAX_ENTRY_ROWS=10000`).
- **Isolated, Resource-Limited SPARQL Execution (`src/app.py`, `src/config.py`, `docs/SPARQL_LIMITS.md`)**: Run client-supplied `/sparql` queries in a separate forked child process on Linux (`PR_SET_PDEATHSIG`, closed inherited FDs, `RLIMIT_AS` memory cap, 10 s end-to-end deadline, 100k row / 32 MiB response caps, bounded concurrency/wait queue, no client caching, and `Retry-After` header preservation). Fails closed with 503 on non-Linux unless `SPARQL_ALLOW_UNISOLATED=true` is explicitly set for local development.
- **`X-Forwarded-Prefix` Validation & Open-Redirect Closure (`src/app.py`, `src/config.py`, `docs/SUBPATH_HOSTING.md`)**: Validate `X-Forwarded-Prefix` as a plain absolute path (with optional `FORWARDED_PREFIX_ALLOWED` allowlist and `TRUST_X_FORWARDED_PREFIX` toggle), disable Starlette's `Host`-header `redirect_slashes`, block dot-segment redirects, re-encode decoded paths, and read raw ASGI `query_string` so `%3F` in paths cannot leak into redirect query strings.
- **Strict SPARQL Input Limits & Decoding (`src/app.py`, `src/config.py`, `docs/SPARQL_LIMITS.md`)**: Enforce `SPARQL_MAX_REQUEST_BYTES` (1 MiB, including chunked bodies) and `SPARQL_MAX_QUERY_LENGTH` (100k chars), reject invalid UTF-8, malformed percent-escapes, lone JSON surrogates, deep JSON nesting, empty queries, and duplicate `query` parameters on both GET and POST.

## [0.5.0] - 2026-11-21

### Added
- **Synchronized `drum-constants` RDF Knowledge Graph & Ontology (`v0.4.0` / `0.5.0`)**: Updated `src/data/codata_constants.ttl` and `src/data/codata.ttl`, and added `--local` synchronization support in `scripts/sync_data.py`.
- **Two-Source Precision, Determination Roles, Formulas & IEEE 754 Metadata (`src/model.py`, `src/app.py`)**:
  - `ConstantValue`: Added `nistValue`, `valueSource`, `adjustmentRole`, `isComputed`, `isNistTruncated`, `significantDigits`, `fitsFloat32`, `fitsFloat64`, and `recommendedNumericType`.
  - `Constant`: Added `nameFr`, `symbolTex`, `formula`, `formulaLatex`, `derivedFrom`, `usedIn` (reverse dependency links), `ucumCode`, and external `identifiers` (`NIST`, `QUDT`, `SI`, `UCUM`).
  - `Concept` & `Quantity`: Added `nameFr`, `definition`, `isRatio`, `isRelationship`, `exactMatch`, and `narrower` concept hierarchy traversal.
- **Rich Query Filtering & NIST Alias Lookup**:
  - Added `?q=`, `?role=`, `?is_exact=`, `?is_computed=`, `?value_source=`, `?recommended_numeric_type=`, `?quantity=`, and `?unit=` filters to `GET /constants`.
  - Enabled lookup by external identifier / NIST symbol alias (e.g., `/constants/hbar`, `/constants/c`) on `GET /constants/{id}` and `GET /constants/{id}/{versionId}`.
- **Read-Only Knowledge Explorer (`GET /explorer`)**: Added an interactive split-pane viewer with KaTeX math rendering, facet filters, formula DAG traversal chips, two-source precision comparison cards, IEEE 754 storage profiles, and 1998–2022 historical timeline tables.
- **Developer Portal & Live Request Sandbox (`GET /`)**: Added an interactive API request builder and multi-language client code generator (`curl`, Python `httpx` + `Decimal`, JavaScript `fetch`).
- **Integrated SPARQL Console & Observability (`GET /sparql`, `GET /health`, `GET /ontology`)**: Integrated browser SPARQL query console directly at `GET /sparql` (when `Accept: text/html` without `?query=`), and added `/health` and content-negotiated `/ontology` endpoints.

### Changed
- **High-Precision Turtle Double Serialization (`src/serializer.py`)**: Ensured `xsd:double` shorthand literals always include a scientific exponent (`e0` appended if absent) so Turtle/N3/TriG parsers round-trip `valueFloat` and `uncertaintyFloat` strictly as `xsd:double`.
- **Fast Graph Traversal & Caching**: Replaced multi-way optional SPARQL scans on REST endpoints with direct indexed rdflib traversal and unconditional `QUERY_CACHE` caching for sub-5ms responses.

### Removed
- Removed the legacy `/playground` application (`src/static/index.html`, `src/static/sparql.html`, `src/static/assets/`) and unmaintained Postman collection links.

## [0.4.1] - 2026-11-02

### Security
- `/sparql` (GET and POST) no longer executes SPARQL `SERVICE` clauses. rdflib evaluates `SERVICE` by sending an HTTP request to the URL in the query, so any caller could make the server contact arbitrary hosts, including internal addresses (server-side request forgery). Queries containing `SERVICE` anywhere in their algebra are now rejected with 400 before evaluation. `SPARQL_LOAD_GRAPHS` is also disabled, so `FROM` / `FROM NAMED` can never fetch remote graphs.

### Fixed
- SPARQL results are evaluated inside the error handler, so evaluation errors return 400 instead of an unhandled 500.
- `format_sparql_results` correctly identifies CONSTRUCT and DESCRIBE queries by query type rather than checking `hasattr(results, 'bindings')`, resolving 500 serializer errors when requesting RDF formats.
- `application/x-www-form-urlencoded` POST queries now include a fallback decoder and explicit `python-multipart` dependency.

### Added
- Tests that `SERVICE` is refused for GET and all three POST content types, including `SERVICE SILENT` and `SERVICE` nested in `OPTIONAL` / `UNION` / `MINUS` / subqueries and in `CONSTRUCT` / `ASK`, verified with a local listener that receives no connection.
- Tests for `CONSTRUCT`, `ASK`, and bound graph namespace prefix resolution on `/sparql`.

## [0.4.0] - 2026-10-20

### Added
- **Default Current Value Filtering**: `GET /constants` now returns only active 2022 CODATA recommended values by default, filtering out discontinued historical constants.
- **Query Parameters**: Added `?include_deprecated=true` to include discontinued constants and `?version=<year>` to retrieve constants published in a specific adjustment year.
- **Typed Response Schemas**: Added Pydantic `response_model` annotations across all FastAPI endpoints (`/concepts`, `/quantities`, `/constants`, `/units`, `/constants/versions`), providing full schema documentation in `/openapi.json` and Swagger UI.
- **JSON-LD `@context`**: Integrated structured `@context` mapping across all `application/ld+json` responses and HTML embedded `<script type="application/ld+json">` blocks.
- **UI Alignments**:
  - `constant.html`: Added deprecation status badge, `nistExponent` display, and `isTruncated` badge.
  - `constants.html`: Added `Deprecated` status badge in the constants table.
  - `unit.html` & `units.html`: Added display of `ucumCode` and `siExpression`.
  - `index.html` (Playground): Aligned detail cards and table rows with `isDeprecated`, `nistExponent`, `ucumCode`, and `siExpression`.
- **Extended Test Suite**: Added 13-test automated pytest suite (`tests/test_fastapi_app.py`) validating OpenAPI contract models, deprecation filtering, SI exact defining and derived relationships, JSON-LD `@context`, unit enrichment, and SPARQL protocol.
- **Licensing Documentation**: Added `LICENSING.md` defining MIT licensing for service source code and CC BY 4.0 for embedded RDF data.

### Changed
- **Decimal Serialization**: Updated `DecimalStr` serializer from `format(x, 'f')` to `str(x)` to preserve scientific notation without generating 34 leading fixed-point zeros in JSON responses.
- **Data Synchronization**: Updated `scripts/sync_data.py` to synchronize both `codata_constants.ttl` and the standalone `codata.ttl` ontology.
- **Model Enhancements**: Added `isDeprecated` to `Constant`, `nistExponent` to `ConstantValue`, and `ucumCode` / `siExpression` to `Unit`.
- **Repository Metadata**: Corrected project governance (`GOVERNANCE.md`) and package descriptions (`pyproject.toml`) to reference the CODATA DRUM Constants API.
- **Documentation**: Updated `README.md` with correct constant identifiers (`SpeedOfLight`, `PlanckConstant`) and `/sparql?query=` parameter usage.

### Removed
- Removed duplicate route definitions for `/constants/versions` and `/constants/versions/{id}` in `src/app.py`.

## [0.3.0] - 2026-02-04

### Added
- Subpath reverse proxy hosting support with `X-Forwarded-Prefix` and `--root-path`.
- In-memory shared read-only RDF graph across Gunicorn / Uvicorn worker processes.
- SPARQL caching for queries taking longer than 0.5 seconds.
- Trailing slash redirection middleware.

### Changed
- Refactored SPARQL queries and UI templates to recognize that `Quantity` is a subclass of `Concept`.

## [0.2.0] - 2025-12-08

### Added
- SPARQL 1.1 protocol endpoint supporting GET and POST (`application/x-www-form-urlencoded`, `application/sparql-query`, `application/json`).
- High-precision custom Turtle, N3, and TriG RDF serializers.
- Interactive SPARQL playground interface (`/playground/sparql`).
- Postman public collection integration.

## [0.1.0] - 2025-08-25

### Added
- Initial FastAPI service implementation for CODATA fundamental physical constants.
- REST endpoints for concepts, quantities, constants, units, and release versions.
- Content negotiation for HTML, JSON, and RDF serializations (Turtle, JSON-LD, RDF/XML, N-Triples, N3, TriG).
- Jinja2 template rendering for browser browsing.
