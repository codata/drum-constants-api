# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
