import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .model import Concept, Quantity, Constant, ConstantValue, Unit, Version, Identifier
from .config import settings
from pydantic import BaseModel
from typing import List, Optional
from decimal import Decimal
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from pathlib import Path
from contextlib import asynccontextmanager
import json


MODEL = Namespace("https://w3id.org/codata/model/")
CONCEPT = Namespace("https://w3id.org/codata/concepts/")
CONSTANT = Namespace("https://w3id.org/codata/constants/")
QUANTITY = Namespace("https://w3id.org/codata/quantities/")
UNIT = Namespace("https://w3id.org/codata/units/")
VERSION = Namespace("https://w3id.org/codata/constants/versions/")

DCTERMS = Namespace("http://purl.org/dc/terms/")
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/vocab/quantitykind/")
UCUM = Namespace("https://w3id.org/uom/")
SICONSTANT = Namespace("https://si-digital-framework.org/constants/")
SIUNIT = Namespace("https://si-digital-framework.org/SI/units/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
WIKIDATA = Namespace("https://www.wikidata.org/entity/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# Global RDF graph
graph: Graph = None

_ext_to_format = {
    ".ttl": "turtle",
    ".rdf": "xml",
    ".owl": "xml",
    ".xml": "xml",
    ".nt": "nt",
    ".n3": "n3",
    ".trig": "trig",
    ".trix": "trix",
    ".jsonld": "json-ld",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize resources on startup, cleanup on shutdown."""
    global graph
    # Startup
    graph = load_rdf_data()
    print(f"Loaded {len(graph)} triples from RDF data files")
    yield
    # Shutdown
    graph = None


ROOT_PATH = os.getenv("ROOT_PATH", "")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    root_path=ROOT_PATH,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Honor X-Forwarded-Prefix for subpath deployments (e.g., /drum)
@app.middleware("http")
async def add_root_path_from_forwarded_prefix(request: Request, call_next):
    prefix = request.headers.get("x-forwarded-prefix")
    if prefix:
        prefix = prefix.rstrip("/")
        if prefix:
            request.scope["root_path"] = prefix
    return await call_next(request)

# Handle trailing slashes by redirecting to non-trailing version
@app.middleware("http")
async def redirect_trailing_slash(request: Request, call_next):
    """Redirect URLs with trailing slashes to their non-trailing-slash version."""
    from fastapi.responses import RedirectResponse
    
    path = request.url.path
    # Skip root path and static files (which may have legitimate trailing slashes)
    if path != "/" and path.endswith("/") and not path.startswith("/playground"):
        # Remove trailing slash and redirect
        new_path = path.rstrip("/")
        new_url = request.url.replace(path=new_path)
        return RedirectResponse(url=new_url, status_code=307)
    
    return await call_next(request)

# Serve SPARQL interface at clean URL without .html extension
# This route must be defined BEFORE mounting static files to avoid being caught by StaticFiles
@app.get("/playground", response_class=FileResponse)
async def playground_root():
    """Serve playground index without losing root_path under proxies."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "index.html")

@app.get("/playground/sparql", response_class=FileResponse)
async def sparql_interface():
    """Serve the SPARQL query interface."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "sparql.html")

# Mount static files for built React frontend (if exists)
# Note: Static files are mounted at /playground relative to the app's actual serving path.
# When deployed with root_path (e.g., /foo), access via: {root_path}/playground
# The templates use {{ root_path }}/playground to generate correct links.
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/playground", StaticFiles(directory=str(static_dir), html=True), name="static")

# Catch-all route for unmatched paths - returns 404 with path and method info
# This must be defined AFTER the StaticFiles mount to catch remaining 404s
@app.get("/{full_path:path}", status_code=404)
async def catch_all_404(full_path: str, request: Request):
    """Catch all 404s not handled by other routes or static files."""
    raise HTTPException(status_code=404, detail=f"Not Found: {full_path}")

# Configure Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


# Custom exception handler for HTTPException to include path for 404 errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException with path information for 404 errors."""
    response_content = {"detail": exc.detail}
    if exc.status_code == 404:
        response_content["path"] = request.url.path
        response_content["method"] = request.method
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content,
    )


class GraphNotInitializedError(Exception):
    """Raised when the RDF graph is not initialized."""

    pass


class SparqlQueryError(Exception):
    """Raised when a SPARQL query fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


def load_rdf_data() -> Graph:
    """Load RDF graph from files in the data directory."""
    g = Graph()
    g.bind("codata", MODEL)
    g.bind("concept", CONCEPT)
    g.bind("constant", CONSTANT)
    g.bind("quantity", QUANTITY)
    g.bind("version", VERSION)
    g.bind("schema", SCHEMA)
    g.bind("si-constant", SICONSTANT)
    g.bind("si-unit", SIUNIT)
    g.bind("skos", SKOS)
    g.bind("unit", UNIT)
    g.bind("ucum", UCUM)
    g.bind("wikidata", WIKIDATA)
    g.bind("xsd", XSD)

    # Data directory is inside src/ (same level as app.py)
    data_dir = settings.DATA_DIR

    if data_dir.exists():
        for f in sorted(data_dir.rglob("*")):
            fmt = _ext_to_format.get(f.suffix.lower())
            if fmt:
                try:
                    g.parse(str(f), format=fmt)
                except Exception as e:
                    print(f"Warning: failed to parse {f} ({fmt}): {e}")
    else:
        print(f"Info: data directory not found at {data_dir}")

    return g


def format_sparql_results(results, accept: str = ""):
    """Format SPARQL query results based on Accept header."""
    # Check if it's a SELECT/ASK query (returns bindings) or CONSTRUCT/DESCRIBE (returns graph)
    if hasattr(results, "bindings"):
        # SELECT or ASK query - return as JSON or XML
        if "application/sparql-results+xml" in accept:
            return Response(
                results.serialize(format="xml"),
                media_type="application/sparql-results+xml",
            )
        # Default: JSON
        return Response(
            results.serialize(format="json"),
            media_type="application/sparql-results+json",
        )
    else:
        # CONSTRUCT or DESCRIBE query - return as RDF
        result_graph = Graph()
        for triple in results:
            result_graph.add(triple)

        if "text/turtle" in accept:
            return Response(
                result_graph.serialize(format="turtle"), media_type="text/turtle"
            )
        if "application/n3" in accept or "text/n3" in accept:
            return Response(result_graph.serialize(format="n3"), media_type="text/n3")
        if "application/ld+json" in accept:
            return Response(
                result_graph.serialize(format="json-ld"),
                media_type="application/ld+json",
            )
        # Default: Turtle
        return Response(
            result_graph.serialize(format="turtle"), media_type="text/turtle"
        )


def run_sparql_query(query: str):
    """Execute a SPARQL query against the global RDF graph. Returns raw results.

    Raises:
        GraphNotInitializedError: If the graph is not initialized.
        SparqlQueryError: If the query fails.
    """
    global graph

    if graph is None:
        raise GraphNotInitializedError("RDF graph not initialized")

    try:
        return graph.query(query)
    except Exception as e:
        raise SparqlQueryError(f"SPARQL query error: {str(e)}", e)


def build_sparql_query(query: str) -> str:
    """Prepopulate a SPARQL query with all defined namespace prefixes.

    Args:
        query_body: The main body of the SPARQL query (without PREFIX declarations)

    Returns:
        Complete SPARQL query string with all prefixes prepended
    """
    prefixes = f"""
PREFIX codata: <{str(MODEL)}>
PREFIX concept: <{str(CONCEPT)}>
PREFIX constant: <{str(CONSTANT)}>
PREFIX quantity: <{str(QUANTITY)}>
PREFIX unit: <{str(UNIT)}>
PREFIX version: <{str(VERSION)}>
PREFIX dcterms: <{str(DCTERMS)}>
PREFIX schema: <{str(SCHEMA)}>
PREFIX qudt: <{str(QUDT)}>
PREFIX ucum: <{str(UCUM)}>
PREFIX si-constant: <{str(SICONSTANT)}>
PREFIX si-unit: <{str(SIUNIT)}>
PREFIX skos: <{str(SKOS)}>
PREFIX wikidata: <{str(WIKIDATA)}>
PREFIX xsd: <{str(XSD)}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
    return prefixes + "\n" + query.strip()


def negotiate_content(uri: str, request: Request, json_response: JSONResponse, 
                     resource_data: dict = None, template_name: str = None) -> Response:
    """Handle content negotiation for a resource URI.
    
    Args:
        uri: The full URI of the resource
        request: The FastAPI request object
        json_response: The JSON response to return if JSON is requested
        resource_data: Dictionary of resource data for HTML template rendering
        template_name: Name of the Jinja2 template to use for HTML rendering
        
    Returns:
        Response in the appropriate format based on Accept header or format query parameter
    """
    # Check for format query parameter (takes precedence over Accept header)
    format_param = request.query_params.get("format", "").lower()
    
    # Map format parameter to appropriate response
    if format_param:
        format_map = {
            "json": "application/json",
            "jsonld": "application/ld+json",
            "turtle": "text/turtle",
            "rdfxml": "application/rdf+xml",
            "ntriples": "application/n-triples",
            "n3": "application/n3",
            "trig": "application/trig"
        }
        # Override accept header if format parameter is valid
        if format_param in format_map:
            accept = format_map[format_param]
        else:
            accept = request.headers.get("accept", "application/json")
    else:
        accept = request.headers.get("accept", "application/json")
    
    # For HTML, render template with JSON-LD embedding
    if resource_data and template_name and ("text/html" in accept or "application/xhtml+xml" in accept):
        # Extract JSON data for JSON-LD script
        json_content = json_response.body.decode('utf-8')
        json_data = json.loads(json_content)
        
        # Pretty-print JSON-LD for embedding
        json_ld = json.dumps(json_data, indent=2)
        
        return templates.TemplateResponse(
            template_name,
            {
                "request": request,
                "resource": resource_data,
                "json_ld": json_ld
            }
        )
    
    # For RDF formats, query the graph and return RDF
    if any(fmt in accept for fmt in ["text/turtle", "application/rdf+xml", "application/n-triples", 
                                      "application/n3", "text/n3", "application/ld+json", "application/trig"]):
        # Query for all triples about this resource
        query = f"""
            CONSTRUCT {{
                <{uri}> ?p ?o .
                ?o ?p2 ?o2 .
            }}
            WHERE {{
                <{uri}> ?p ?o .
                OPTIONAL {{ ?o ?p2 ?o2 }}
            }}
            """
        query = build_sparql_query(query)
        result_graph = run_sparql_query(query).graph
        
        # Return appropriate RDF format based on Accept header
        if "application/rdf+xml" in accept or "application/xml" in accept:
            return Response(
                result_graph.serialize(format="xml"), media_type="application/rdf+xml"
            )
        if "application/n-triples" in accept:
            return Response(
                result_graph.serialize(format="nt"), media_type="application/n-triples"
            )
        if "text/turtle" in accept:
            return Response(
                result_graph.serialize(format="turtle"), media_type="text/turtle"
            )
        if "application/n3" in accept or "text/n3" in accept:
            return Response(result_graph.serialize(format="n3"), media_type="text/n3")
        if "application/ld+json" in accept:
            return Response(
                result_graph.serialize(format="json-ld"),
                media_type="application/ld+json",
            )
        if "application/trig" in accept:
            return Response(
                result_graph.serialize(format="trig"), media_type="application/trig"
            )
        # Default: Turtle
        return Response(
            result_graph.serialize(format="turtle"), media_type="text/turtle"
        )
    
    # Default: Return JSON
    return json_response


@app.get("/")
async def index():
    return {"message": "Welcome to CODATA DRUM Physical Fundamental Constants API!"}


@app.get("/concepts")
async def concepts(request: Request):
    """Retrieve all concepts using SPARQL and return as structured data.
    
    This includes both pure Concepts and Quantities (which are also Concepts).
    """
    try:
        # Query for both Concepts and Quantities (which are also Concepts)
        query = """
            SELECT ?uri ?label ?type
            WHERE {
                {
                    ?uri a codata:Concept ;
                        skos:prefLabel ?label .
                    FILTER (lang(?label) = "" || lang(?label) = "en")
                    BIND ("Concept" AS ?type)
                }
                UNION
                {
                    ?uri a codata:Quantity ;
                        skos:prefLabel ?label .
                    FILTER (lang(?label) = "" || lang(?label) = "en")
                    BIND ("Quantity" AS ?type)
                }
            }
            ORDER BY ?label
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)

        # Build Concept instances with computed isQuantity attribute
        concepts_list = []
        for row in bindings:
            c_uri = str(row.uri)
            c_id = c_uri.split("/")[-1]
            c_label = str(row.label) if row.label else c_id
            is_quantity = str(row.type) == "Quantity"
            concept_dict = Concept(id=c_id, uri=c_uri, name=c_label).model_dump(exclude_none=True)
            concept_dict["isQuantity"] = is_quantity
            concepts_list.append(concept_dict)

        # Check if HTML is requested
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            json_ld = json.dumps(concepts_list, indent=2)
            return templates.TemplateResponse(
                "concepts.html",
                {"request": request, "resources": concepts_list, "json_ld": json_ld}
            )

        return JSONResponse(content=concepts_list)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/concepts/{id}")
async def concept(id: str, request: Request):
    """Retrieve concept information with content negotiation support.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        concept_uri = CONCEPT[id]
        quantity_uri = QUANTITY[id]
        
        # SPARQL query to get concept details and related concepts
        # Check both concept and quantity namespaces, as a Quantity can also be a Concept
        query = f"""
            SELECT ?uri ?label ?part ?broader ?isQuantity
            WHERE {{
                {{
                    BIND (<{concept_uri}> AS ?uri)
                    <{concept_uri}> a codata:Concept ;
                        skos:prefLabel ?label .
                    FILTER (lang(?label) = "" || lang(?label) = "en")
                    OPTIONAL {{ <{concept_uri}> dcterms:hasPart ?part }}
                    OPTIONAL {{ <{concept_uri}> skos:broader ?broader }}
                    BIND (false AS ?isQuantity)
                }}
                UNION
                {{
                    BIND (<{quantity_uri}> AS ?uri)
                    <{quantity_uri}> a codata:Quantity ;
                        skos:prefLabel ?label .
                    FILTER (lang(?label) = "" || lang(?label) = "en")
                    OPTIONAL {{ <{quantity_uri}> dcterms:hasPart ?part }}
                    OPTIONAL {{ <{quantity_uri}> skos:broader ?broader }}
                    BIND (true AS ?isQuantity)
                }}
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        # Extract results
        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"Concept '{id}' not found")

        # Get label and URI from first result
        first = bindings[0]
        uri = str(first.uri) if first.uri else str(concept_uri)
        label = str(first.label) if first.label else ""

        # Check if this is a Quantity (found in quantity namespace)
        is_quantity = bool(first.isQuantity) if hasattr(first, 'isQuantity') and first.isQuantity else False

        # Collect unique parts (sub-concepts) from all results
        parts = []
        seen_parts = set()

        # Collect unique broader concepts from all results
        broader_concepts = []
        seen_broader = set()

        for row in bindings:
            if row.part and str(row.part) not in seen_parts:
                part_uri = str(row.part)
                part_id = part_uri.split("/")[-1]
                parts.append(Concept(id=part_id, uri=part_uri, name=part_id))
                seen_parts.add(part_uri)
            if row.broader and str(row.broader) not in seen_broader:
                broader_uri = str(row.broader)
                broader_id = broader_uri.split("/")[-1]
                broader_concepts.append(Concept(id=broader_id, uri=broader_uri, name=broader_id))
                seen_broader.add(broader_uri)

        # If this concept is a Quantity, set quantity field; otherwise query for related quantities
        concept_quantity = None
        concept_quantities = None

        if is_quantity:
            # This concept is itself a Quantity
            concept_quantity = Quantity(id=id, uri=uri, name=label)
        else:
            # Query for quantities that have dcterms:hasPart pointing to this concept
            quantities_query = f"""
                SELECT ?quantityUri ?quantityLabel
                WHERE {{
                    ?quantityUri a codata:Quantity ;
                        dcterms:hasPart <{uri}> ;
                        skos:prefLabel ?quantityLabel .
                    FILTER (lang(?quantityLabel) = "" || lang(?quantityLabel) = "en")
                }}
                """
            quantities_query = build_sparql_query(quantities_query)
            quantities_results = run_sparql_query(quantities_query)
            quantities_bindings = list(quantities_results)

            # Build Quantity instances
            if quantities_bindings:
                concept_quantities = []
                for row in quantities_bindings:
                    q_uri = str(row.quantityUri)
                    q_id = q_uri.split("/")[-1]
                    q_label = str(row.quantityLabel) if row.quantityLabel else q_id
                    concept_quantities.append(Quantity(id=q_id, uri=q_uri, name=q_label))

        # Populate Concept from SPARQL results
        concept_obj = Concept(
            id=id,
            uri=uri,
            name=label,
            parts=parts if parts else None,
            broader=broader_concepts if broader_concepts else None,
            quantity=concept_quantity,
            quantities=concept_quantities,
        )

        json_response = JSONResponse(content=concept_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = concept_obj.model_dump(exclude_none=True)
        
        return negotiate_content(uri, request, json_response, resource_data, "concept.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/quantities")
async def quantities(request: Request):
    """Retrieve all quantities using SPARQL and return as structured data."""
    try:
        query = """
            SELECT ?uri ?label ?constantUri ?constantLabel
            WHERE {
                ?uri a codata:Quantity ;
                    skos:prefLabel ?label .
                FILTER (lang(?label) = "" || lang(?label) = "en")
                OPTIONAL {
                    ?uri codata:hasConstant ?constantUri .
                    ?constantUri skos:prefLabel ?constantLabel .
                    FILTER (lang(?constantLabel) = "" || lang(?constantLabel) = "en")
                }
            }
            ORDER BY ?label
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)

        # Build Quantity instances with their constants
        quantities_dict = {}
        for row in bindings:
            q_uri = str(row.uri)
            q_id = q_uri.split("/")[-1]
            
            # Skip if we already processed this quantity
            if q_id not in quantities_dict:
                q_label = str(row.label) if row.label else q_id
                quantities_dict[q_id] = {
                    "quantity": Quantity(id=q_id, uri=q_uri, name=q_label),
                    "constants": []
                }
            
            # Add constant if present
            if row.constantUri:
                c_uri = str(row.constantUri )
                c_id = c_uri.split("/")[-1]
                c_label = str(row.constantLabel) if row.constantLabel else c_id
                quantities_dict[q_id]["constants"].append(
                    Constant(id=c_id, uri=c_uri, name=c_label)
                )
        
        # Build final list with constants
        quantities_list = []
        for q_data in quantities_dict.values():
            quantity = q_data["quantity"]
            if q_data["constants"]:
                quantity.constants = q_data["constants"]
            quantities_list.append(quantity)

        # Prepare JSON serializable data
        quantities_json = [q.model_dump(exclude_none=True) for q in quantities_list]
        
        # Check if HTML is requested
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            json_ld = json.dumps(quantities_json, indent=2)
            return templates.TemplateResponse(
                "quantities.html",
                {"request": request, "resources": quantities_json, "json_ld": json_ld}
            )

        return JSONResponse(content=quantities_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/quantities/{id}")
async def quantity(id: str, request: Request):
    """Retrieve quantity information with content negotiation support.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        uri = QUANTITY[id]
        # SPARQL query to get quantity details, concepts, and constants
        query = f"""
            SELECT ?label ?concept ?constant
            WHERE {{
                <{uri}> skos:prefLabel ?label .
                OPTIONAL {{ <{uri}> dcterms:hasPart ?concept }}
                OPTIONAL {{ <{uri}> codata:hasConstant ?constant }}
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        # Extract results
        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"Quantity '{id}' not found")

        # Get label from first result
        first = bindings[0]
        label = str(first.label) if first.label else ""

        # Collect unique concepts and constants from all results
        concepts = []
        seen_concepts = set()
        constants = []
        seen_constants = set()

        for row in bindings:
            if row.concept and str(row.concept) not in seen_concepts:
                concept_uri = str(row.concept)
                concept_id = concept_uri.split("/")[-1]
                concepts.append(Concept(id=concept_id, uri=concept_uri, name=concept_id))
                seen_concepts.add(concept_uri)
            if row.constant and str(row.constant) not in seen_constants:
                constant_uri = str(row.constant)
                constant_id = constant_uri.split("/")[-1]
                constants.append(Constant(id=constant_id, uri=constant_uri, name=constant_id))
                seen_constants.add(constant_uri)

        # Populate Quantity from SPARQL results
        quantity_obj = Quantity(
            id=id,
            uri=str(uri),
            name=label,
            concepts=concepts if concepts else None,
            constants=constants if constants else None,
        )

        json_response = JSONResponse(content=quantity_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = quantity_obj.model_dump(exclude_none=True)
        
        return negotiate_content(str(uri), request, json_response, resource_data, "quantity.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/units")
async def units(request: Request):
    """Retrieve all units using SPARQL and return as structured data."""
    try:
        # Query for units - get the literal identifier as the name
        query = """
            SELECT ?uri ?name
            WHERE {
                ?uri a codata:Unit ;
                    schema:identifier ?name .
                FILTER (isLiteral(?name))
            }
            ORDER BY ?name
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)

        # Build Unit instances
        units_list = []
        for row in bindings:
            u_uri = str(row.uri)
            u_id = u_uri.split("/")[-1]
            u_name = str(row.name) if row.name else u_id
            units_list.append(Unit(id=u_id, uri=u_uri, name=u_name))

        # Prepare JSON serializable data
        units_json = [u.model_dump(exclude_none=True) for u in units_list]
        
        # Check if HTML is requested
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            json_ld = json.dumps(units_json, indent=2)
            return templates.TemplateResponse(
                "units.html",
                {"request": request, "resources": units_json, "json_ld": json_ld}
            )

        return JSONResponse(content=units_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")

@app.get("/constants")
async def constants(request: Request):
    """Retrieve all constants using SPARQL and return as structured data."""
    try:
        query = """
            SELECT ?uri ?label ?quantityUri ?quantityLabel ?valueUri ?value ?uncertainty ?versionId ?unitUri
            WHERE {
                ?uri a codata:Constant ;
                    skos:prefLabel ?label .
                FILTER (lang(?label) = "" || lang(?label) = "en")
                OPTIONAL { 
                    ?uri codata:hasQuantity ?quantityUri .
                    ?quantityUri skos:prefLabel ?quantityLabel .
                    FILTER (lang(?quantityLabel) = "" || lang(?quantityLabel) = "en")
                }
                OPTIONAL {
                    ?uri codata:hasUnit ?unitUri .
                }
                OPTIONAL {
                    ?uri codata:hasValue ?valueUri .
                    ?valueUri codata:value ?value ;
                              codata:versionId ?versionId .
                    OPTIONAL { ?valueUri codata:uncertainty ?uncertainty }
                }
            }
            ORDER BY ?label DESC(?versionId)
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)

        # Build Constant instances - take first row per constant (highest versionId due to ORDER BY DESC)
        constants_dict = {}
        for row in bindings:
            c_uri = str(row.uri)
            c_id = c_uri.split("/")[-1]
            
            # Skip if we already processed this constant (we already have the latest value)
            if c_id in constants_dict:
                continue
                
            c_label = str(row.label) if row.label else c_id
            
            # Build quantity if present
            quantity = None
            if row.quantityUri:
                q_uri = str(row.quantityUri)
                q_id = q_uri.split("/")[-1]
                q_label = str(row.quantityLabel) if row.quantityLabel else q_id
                quantity = Quantity(id=q_id, uri=q_uri, name=q_label)
            
            # Build unit if present
            unit = None
            if row.unitUri:
                u_uri = str(row.unitUri)
                u_id = u_uri.split("/")[-1]
                unit = Unit(id=u_id, uri=u_uri)
            
            # Build latest constant value if present (simplified for list view)
            latest_value = None
            if row.value:
                value_uri = str(row.valueUri) if row.valueUri else ""
                value_id = value_uri.split("/")[-1] if value_uri else "unknown"
                version_id = str(row.versionId) if row.versionId else "unknown"
                latest_value = ConstantValue(
                    uri=value_uri,
                    id=value_id,
                    value=Decimal(str(row.value)),
                    uncertainty=Decimal(str(row.uncertainty)) if row.uncertainty else Decimal("0"),
                    versionId=version_id,
                )
            
            constant = Constant(
                id=c_id,
                uri=c_uri,
                name=c_label,
                quantity=quantity,
                value=latest_value,
                unit=unit,
            )
            constants_dict[c_id] = constant

        constants_list = [c.model_dump(exclude_none=True) for c in constants_dict.values()]
        
        # Check if HTML is requested
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            json_ld = json.dumps(constants_list, indent=2)
            return templates.TemplateResponse(
                "constants.html",
                {"request": request, "resources": constants_list, "json_ld": json_ld}
            )
        
        return JSONResponse(content=constants_list)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/{id}")
async def constant(id: str, request: Request):
    """Retrieve constant information with content negotiation support.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        uri = CONSTANT[id]
        # SPARQL query to get constant details, quantity, and unit
        query = f"""
            SELECT ?label ?quantity ?unit
            WHERE {{
                <{uri}> skos:prefLabel ?label .
                FILTER (lang(?label) = "" || lang(?label) = "en")
                OPTIONAL {{ <{uri}> codata:hasQuantity ?quantity }}
                OPTIONAL {{ <{uri}> codata:hasUnit ?unit }}
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        # Extract results
        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"Constant '{id}' not found")

        # Get label from first result
        first = bindings[0]
        label = str(first.label) if first.label else ""

        # Get quantity (should be same across all rows)
        quantity = None
        if first.quantity:
            quantity_uri = str(first.quantity)
            quantity_id = quantity_uri.split("/")[-1]
            quantity = Quantity(id=quantity_id, uri=quantity_uri, name=quantity_id)

        # Get unit (should be same across all rows)
        unit = None
        if first.unit:
            unit_uri = str(first.unit)
            unit_id = unit_uri.split("/")[-1]
            unit = Unit(id=unit_id, uri=unit_uri)

        # Query for all ConstantValues that reference this constant via dcterms:isVersionOf
        values_query = f"""
            SELECT ?valueUri ?val ?uncertainty ?isExact ?isTruncated ?versionId ?versionUri
            WHERE {{
                ?valueUri a codata:ConstantValue ;
                    dcterms:isVersionOf <{uri}> ;
                    codata:value ?val ;
                    codata:isExact ?isExact ;
                    codata:isTruncated ?isTruncated ;
                    codata:versionId ?versionId ;
                    codata:hasVersion ?versionUri .
                OPTIONAL {{ ?valueUri codata:uncertainty ?uncertainty }}
            }}
            ORDER BY DESC(?versionId)
            """
        values_query = build_sparql_query(values_query)
        values_results = run_sparql_query(values_query)
        values_bindings = list(values_results)

        # Build ConstantValue instances
        constant_values: list[ConstantValue] = []
        for row in values_bindings:
            value_uri = str(row.valueUri)
            value_id = value_uri.split("/")[-1]
            version_uri = str(row.versionUri)
            version_id = version_uri.split("/")[-1]

            cv = ConstantValue(
                id=value_id,
                uri=value_uri,
                value=Decimal(str(row.val)),
                uncertainty=Decimal(str(row.uncertainty)) if hasattr(row, 'uncertainty') and row.uncertainty else None,
                isExact=bool(row.isExact),
                isTruncated=bool(row.isTruncated),
                versionId=str(row.versionId),
                version=Version(id=version_id, uri=version_uri),
            )
            constant_values.append(cv)

        # The first value (highest versionId due to ORDER BY DESC) is the current value
        # The rest are historical values
        current_value = constant_values[0] if constant_values else None
        historical_values = constant_values[1:] if len(constant_values) > 1 else None

        # Populate Constant from SPARQL results
        constant_obj = Constant(
            id=id,
            uri=str(uri),
            name=label,
            quantity=quantity,
            unit=unit,
            value=current_value,
            historicalValues=historical_values,
        )

        json_response = JSONResponse(content=constant_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = constant_obj.model_dump(exclude_none=True)
        
        return negotiate_content(str(uri), request, json_response, resource_data, "constant.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/{id}/{versionId}")
async def constant_value(id: str, versionId: str, request: Request):
    """Retrieve a specific ConstantValue for a constant in a given CODATA version.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        constant_uri = CONSTANT[id]
        version_uri = VERSION[versionId]
        
        # SPARQL query to get the specific ConstantValue for this constant and version
        query = f"""
            SELECT ?valueUri ?val ?uncertainty ?isExact ?isTruncated ?label ?quantity ?unit
            WHERE {{
                ?valueUri a codata:ConstantValue ;
                    dcterms:isVersionOf <{constant_uri}> ;
                    codata:value ?val ;
                    codata:isExact ?isExact ;
                    codata:isTruncated ?isTruncated ;
                    codata:hasVersion <{version_uri}> .
                OPTIONAL {{ ?valueUri codata:uncertainty ?uncertainty }}
                OPTIONAL {{ <{constant_uri}> skos:prefLabel ?label }}
                OPTIONAL {{ <{constant_uri}> codata:hasQuantity ?quantity }}
                OPTIONAL {{ <{constant_uri}> codata:hasUnit ?unit }}
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)
        
        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"ConstantValue for constant '{id}' in version '{versionId}' not found")
        
        first = bindings[0]
        value_uri = str(first.valueUri)  # type: ignore
        value_id = value_uri.split("/")[-1]
        
        # Get constant label
        label = str(first.label) if first.label else ""  # type: ignore
        
        # Get quantity
        quantity = None
        if first.quantity:  # type: ignore
            quantity_uri = str(first.quantity)  # type: ignore
            quantity_id = quantity_uri.split("/")[-1]
            quantity = Quantity(id=quantity_id, uri=quantity_uri, name=quantity_id)
        
        # Get unit
        unit = None
        if first.unit:  # type: ignore
            unit_uri = str(first.unit)  # type: ignore
            unit_id = unit_uri.split("/")[-1]
            unit = Unit(id=unit_id, uri=unit_uri)
        
        # Create the ConstantValue object
        constant_value_obj = ConstantValue(
            id=value_id,
            uri=value_uri,
            value=Decimal(str(first.val)),  # type: ignore
            uncertainty=Decimal(str(first.uncertainty)) if hasattr(first, 'uncertainty') and first.uncertainty else None,  # type: ignore
            isExact=bool(first.isExact),  # type: ignore
            isTruncated=bool(first.isTruncated),  # type: ignore
            versionId=versionId,
            version=Version(id=versionId, uri=str(version_uri)),
        )
        
        # Create a Constant wrapper with this specific value for HTML template
        constant_obj = Constant(
            id=id,
            uri=str(constant_uri),
            name=label,
            quantity=quantity,
            unit=unit,
            value=constant_value_obj,
        )
        
        json_response = JSONResponse(content=constant_value_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = constant_obj.model_dump(exclude_none=True)
        
        return negotiate_content(value_uri, request, json_response, resource_data, "constant.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/units/{id}")
async def unit(id: str, request: Request):
    """Retrieve unit information with content negotiation support.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        uri = UNIT[id]
        
        # SPARQL query to get unit identifiers
        # Units have schema:identifier which can be:
        # - A literal string (the unit symbol itself)
        # - A URI to a PropertyValue (for SI/UCUM identifiers)
        query = f"""
            SELECT ?identifier
            WHERE {{
                <{uri}> a codata:Unit ;
                    schema:identifier ?identifier .
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"Unit '{id}' not found")

        # Process identifiers - separate URIs (PropertyValues) from literals
        identifier_uris = []
        unit_symbol = None
        
        for row in bindings:
            identifier = row.identifier
            if isinstance(identifier, URIRef):
                identifier_uris.append(str(identifier))
            elif isinstance(identifier, Literal):
                unit_symbol = str(identifier)

        # Query PropertyValue details for each identifier URI
        identifiers = []
        if identifier_uris:
            uris_values = " ".join(f"<{u}>" for u in identifier_uris)
            pv_query = f"""
                SELECT ?pvUri ?pvId ?propertyId ?url
                WHERE {{
                    VALUES ?pvUri {{ {uris_values} }}
                    ?pvUri a schema:PropertyValue ;
                        schema:identifier ?pvId ;
                        schema:propertyID ?propertyId .
                    OPTIONAL {{ ?pvUri schema:url ?url }}
                }}
                """
            pv_query = build_sparql_query(pv_query)
            pv_results = run_sparql_query(pv_query)
            pv_bindings = list(pv_results)

            for row in pv_bindings:
                pv_uri = str(row.pvUri)
                pv_id = str(row.propertyId) if row.propertyId else pv_uri.split("#")[-1]
                pv_value = str(row.pvId) if row.pvId else None
                pv_url = str(row.url) if row.url else None
                
                identifiers.append(Identifier(
                    id=pv_id,
                    value=pv_value,
                    url=pv_url,
                ))

        # Query for constants that have this unit
        constants_query = f"""
            SELECT ?constantUri ?constantLabel
            WHERE {{
                ?constantUri a codata:Constant ;
                    codata:hasUnit <{uri}> ;
                    skos:prefLabel ?constantLabel .
                FILTER (lang(?constantLabel) = "" || lang(?constantLabel) = "en")
            }}
            """
        constants_query = build_sparql_query(constants_query)
        constants_results = run_sparql_query(constants_query)
        constants_bindings = list(constants_results)

        # Build Constant instances
        constants = []
        for row in constants_bindings:
            c_uri = str(row.constantUri)
            c_id = c_uri.split("/")[-1]
            c_label = str(row.constantLabel) if row.constantLabel else c_id
            constants.append(Constant(id=c_id, uri=c_uri, name=c_label))

        # Build the Unit response
        unit_obj = Unit(
            id=id,
            uri=str(uri),
            name=unit_symbol,
            identifiers=identifiers if identifiers else None,
            constants=constants if constants else None,
        )

        json_response = JSONResponse(content=unit_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = unit_obj.model_dump(exclude_none=True)
        
        return negotiate_content(str(uri), request, json_response, resource_data, "unit.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/versions")
async def versions(request: Request):
    """Retrieve all versions using SPARQL and return as structured data."""
    try:
        query = """
            SELECT ?uri ?identifier ?issued
            WHERE {
                ?uri a codata:Version ;
                    schema:identifier ?identifier .
                OPTIONAL { ?uri dcterms:issued ?issued }
            }
            ORDER BY ?identifier
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)

        # Build Version instances
        versions_list = []
        for row in bindings:
            v_uri = str(row.uri)
            v_id = str(row.identifier) if row.identifier else v_uri.split("/")[-1]
            
            # Parse the issued date
            published = None
            if row.issued:
                from datetime import date
                published = date.fromisoformat(str(row.issued))
            
            versions_list.append(Version(id=v_id, uri=v_uri, published=published))

        # Prepare JSON serializable data
        versions_json = [v.model_dump(exclude_none=True) for v in versions_list]
        
        # Check if HTML is requested
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            json_ld = json.dumps(versions_json, indent=2)
            return templates.TemplateResponse(
                "versions.html",
                {"request": request, "resources": versions_json, "json_ld": json_ld}
            )
        
        return JSONResponse(content=versions_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")

@app.get("/versions/{id}")
async def version(id: str, request: Request):
    """Retrieve version information with content negotiation support.
    
    Returns JSON by default, or RDF formats based on Accept header.
    """
    try:
        uri = VERSION[id]
        
        # SPARQL query to get version details
        query = f"""
            SELECT ?issued ?identifier
            WHERE {{
                <{uri}> a codata:Version .
                OPTIONAL {{ <{uri}> dcterms:issued ?issued }}
                OPTIONAL {{ <{uri}> schema:identifier ?identifier }}
            }}
            """
        query = build_sparql_query(query)
        results = run_sparql_query(query)

        bindings = list(results)
        if not bindings:
            raise HTTPException(status_code=404, detail=f"Version '{id}' not found")

        first = bindings[0]
        
        # Parse the issued date
        published = None
        if first.issued:
            from datetime import date
            # The date is in xsd:date format (YYYY-MM-DD)
            published = date.fromisoformat(str(first.issued))

        # Query for constants that have ConstantValues belonging to this version
        # ConstantValue has codata:hasVersion pointing to the version
        # and dcterms:isVersionOf pointing to the parent Constant
        constants_query = f"""
            SELECT DISTINCT ?constantUri ?constantLabel
            WHERE {{
                ?constantValue a codata:ConstantValue ;
                    codata:hasVersion <{uri}> ;
                    dcterms:isVersionOf ?constantUri .
                ?constantUri skos:prefLabel ?constantLabel .
                FILTER (lang(?constantLabel) = "" || lang(?constantLabel) = "en")
            }}
            ORDER BY ?constantLabel
            """
        constants_query = build_sparql_query(constants_query)
        constants_results = run_sparql_query(constants_query)
        constants_bindings = list(constants_results)

        # Build Constant instances
        constants = []
        for row in constants_bindings:
            c_uri = str(row.constantUri)
            c_id = c_uri.split("/")[-1]
            c_label = str(row.constantLabel) if row.constantLabel else c_id
            constants.append(Constant(id=c_id, uri=c_uri, name=c_label))

        # Build the Version response
        version_obj = Version(
            id=id,
            uri=str(uri),
            published=published,
            constants=constants if constants else None,
        )

        json_response = JSONResponse(content=version_obj.model_dump(exclude_none=True))
        
        # Prepare data for HTML template
        resource_data = version_obj.model_dump(exclude_none=True)
        
        return negotiate_content(str(uri), request, json_response, resource_data, "version.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/sparql")
async def sparql_get(query: str, request: Request):
    """Execute a SPARQL query via GET request (SPARQL 1.1 Protocol §2.1.1).

    Query parameters:
        query: The SPARQL query string (required)
        default-graph-uri: Default graph URI (optional, not implemented)
        named-graph-uri: Named graph URI (optional, not implemented)
    """
    try:
        results = run_sparql_query(query)
    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=e.message)

    accept = request.headers.get("accept", "")
    return format_sparql_results(results, accept)


@app.post("/sparql")
async def sparql_post(request: Request):
    """Execute a SPARQL query via POST request (SPARQL 1.1 Protocol §2.1.2 and §2.1.3).

    Supports three content types per SPARQL Protocol:
    1. application/x-www-form-urlencoded - URL-encoded parameters with 'query' field
    2. application/sparql-query - Direct unencoded SPARQL query string as body
    3. application/json - JSON body with 'query' field (extension for convenience)
    """
    content_type = request.headers.get("content-type", "")

    # Extract query based on content type
    if "application/x-www-form-urlencoded" in content_type:
        # SPARQL Protocol §2.1.2: query via POST with URL-encoded parameters
        form_data = await request.form()
        query = form_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' parameter")
    elif "application/sparql-query" in content_type:
        # SPARQL Protocol §2.1.3: query via POST directly
        body = await request.body()
        query = body.decode("utf-8")
        if not query:
            raise HTTPException(status_code=400, detail="Empty query body")
    elif "application/json" in content_type:
        # Extension: JSON body for convenience (not in SPARQL Protocol spec)
        try:
            json_body = await request.json()
            query = json_body.get("query")
            if not query:
                raise HTTPException(
                    status_code=400, detail="Missing 'query' field in JSON body"
                )
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type. Use application/x-www-form-urlencoded, "
            "application/sparql-query, or application/json",
        )

    try:
        results = run_sparql_query(query)
    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=e.message)

    accept = request.headers.get("accept", "")
    return format_sparql_results(results, accept)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


