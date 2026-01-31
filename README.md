# Drum API

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

**⚠️ EVALUATION PURPOSES ONLY ⚠️**

This is an **experimental edition** of the CODATA DRUM API. It is provided for evaluation and feedback purposes only. Features, endpoints, and data structures are subject to change without notice.

## Overview

The Drum API is a semantic gateway for browsing CODATA fundamental physical constants. Built with FastAPI and backed by an RDF knowledge graph, it provides rich semantic data with full content negotiation and **high-precision serialization**.

The underlying data is sourced from the [CODATA DRUM Constants](https://github.com/codata/drum-constants) repository.

## Try It Now

The API is available for testing at:
**[https://api.codata.org/drum](https://api.codata.org/drum)**

### Interactive Documentation & Tools
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **Postman**: [CODATA DRUM Public Collection](https://www.postman.com/codata-org/workspace/codata-drum/collection/41633755-4a0c7834-ce9a-4d88-b4f0-c88f242094f5)
- **SPARQL Playground**: `/playground/sparql`

## Features
- **RESTful API** - High-performance engine built with FastAPI (Python 3.11+)
- **High-Precision RDF** - Custom serializers preserving full decimal precision for physical constants
- **Content Negotiation** - HTML, JSON, and RDF formats (Turtle, N-Triples, N3, JSON-LD, RDF/XML, TriG)
- **SPARQL Engine** - Direct access to the RDF knowledge graph
- **Linked Data** - 100% URL-resolvable resources with rich semantic relationships
- **Interactive Tools** - Integrated Swagger, ReDoc, and Postman Collection
- **Modern Stack** - Pydantic V2, Hatch project management, and async processing

## API Endpoints

### Resource Lists
- `GET /` - API welcome and overview
- `GET /concepts` - Browse all concepts (including quantities)
- `GET /quantities` - Browse all physical quantities
- `GET /constants` - Browse all fundamental constants
- `GET /units` - Browse all units of measurement
- `GET /constants/versions` - Browse all CODATA version releases

### Resource Details
- `GET /concepts/{id}` - Get concept details with relationships
- `GET /quantities/{id}` - Get quantity details with associated constants
- `GET /constants/{id}` - Get constant details with values across versions
- `GET /units/{id}` - Get unit details
- `GET /constants/versions/{id}` - Get version release information

### Advanced
- `GET /sparql` - Direct SPARQL query endpoint (query parameter: `q`)

All endpoints support content negotiation via `Accept` header or `?format=` query parameter.

## Quickstart

### Prerequisites
- Python 3.11 or higher
- [Hatch](https://hatch.pypa.io/) for project management

### Setup and Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd drum-api
   ```

2. Create the development environment:
   ```bash
   hatch env create
   ```

3. Run the API server:
   ```bash
   hatch run api
   ```

4. Visit the API:
   - HTML Browser: http://localhost:8000/
   - API Documentation: http://localhost:8000/docs
   - Browse Constants: http://localhost:8000/constants

5. Run tests:
   ```bash
   hatch run test
   ```

## API Documentation
When running the API, automatic documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Content Negotiation

All resource detail endpoints support multiple formats via the `Accept` header or `?format=` query parameter:

### Supported Formats

| Format | Accept Header | Query Parameter | Use Case |
|--------|--------------|-----------------|----------|
| HTML | `text/html` | `?format=html` | Human-readable browsing |
| JSON | `application/json` | `?format=json` | API integration (default) |
| JSON-LD | `application/ld+json` | `?format=jsonld` | Linked data / semantic web |
| Turtle | `text/turtle` | `?format=turtle` | RDF serialization |
| RDF/XML | `application/rdf+xml` | `?format=rdfxml` | RDF serialization |
| N-Triples | `application/n-triples` | `?format=ntriples` | RDF serialization |
| N3 | `text/n3` | `?format=n3` | RDF serialization |
| TriG | `application/trig` | `?format=trig` | RDF dataset serialization |

### Examples

Using Accept header:
```bash
# Get speed of light constant as Turtle
curl -H "Accept: text/turtle" http://localhost:8000/constants/speed-of-light-in-vacuum

# Get Planck constant as JSON-LD
curl -H "Accept: application/ld+json" http://localhost:8000/constants/planck-constant
```

Using query parameter:
```bash
# Get speed of light constant as Turtle
curl http://localhost:8000/constants/speed-of-light-in-vacuum?format=turtle

# Get Planck constant as JSON-LD
curl http://localhost:8000/constants/planck-constant?format=jsonld
```

### HTML Format Selector
When viewing resources in HTML format, a format selector in the top-right corner allows switching between all available formats.

## Project Structure
```
drum-api/
├── src/
│   ├── app.py              # FastAPI application with endpoints
│   ├── model.py            # Pydantic data models
│   ├── data/               # RDF data files (.ttl, .rdf)
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # Static assets (CSS, JS)
├── tests/
│   └── test_fastapi_app.py # API test suite
├── pyproject.toml          # Project configuration and dependencies
├── gunicorn.conf.py        # Gunicorn production configuration
└── README.md
```

## Development

### Running with Uvicorn

The recommended way to run the API during development:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Running Tests

```bash
# Run tests with Hatch
hatch run test

# Or use pytest directly
pytest tests/
```

## Production Deployment

### Using Uvicorn with Workers

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn + Uvicorn Workers

For high-performance production deployment:

```bash
gunicorn src.app:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --preload --bind 0.0.0.0:8100 --access-logfile - --error-logfile - --log-level info
```

Configuration is provided in `gunicorn.conf.py`.

### Subpath Hosting

The API fully supports deployment under a subpath (e.g., `https://api.example.org/drum-api/`).

**Method 1: Using `--root-path` flag**

```bash
uvicorn src.app:app --root-path /drum-api --host 0.0.0.0 --port 8000
```

Then access at: `http://localhost:8000/drum-api/`

**Method 2: Behind Nginx reverse proxy**

```nginx
location /drum-api/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Run the app normally:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Then access at: `https://yourdomain.com/drum-api/`

**Method 3: Behind Apache reverse proxy**

```apache
<Location /drum-api>
    ProxyPass http://localhost:8000/
    ProxyPassReverse http://localhost:8000/
    RequestHeader set X-Forwarded-Prefix /drum-api
</Location>
```

All HTML navigation, API endpoints, and content negotiation work correctly under subpaths.

## Data Sources

This API serves fundamental physical constants from:
- **CODATA** - Committee on Data of the International Science Council
- RDF knowledge graph with semantic relationships
- Constants include values, uncertainties, and version history

## Technology Stack

- **FastAPI** - Modern Python web framework
- **RDFLib** - RDF graph database and SPARQL queries
- **Pydantic** - Data validation and serialization
- **Jinja2** - HTML templating
- **Hatch** - Project management and packaging
- **Uvicorn** - ASGI server

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please ensure:
- Tests pass: `hatch run test`
- Code follows existing patterns
- API changes are documented

## License

MIT
