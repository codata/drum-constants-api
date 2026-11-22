from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, List, Optional
import urllib.parse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.plugin import register
import rdflib.plugins.sparql
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.serializer import Serializer
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import settings
from .model import Concept, Constant, ConstantValue, Identifier, Quantity, Unit, Version

# Never fetch remote graphs named in FROM / FROM NAMED
rdflib.plugins.sparql.SPARQL_LOAD_GRAPHS = False

# Register custom high-precision serializers
register("turtle-hp", Serializer, "src.serializer", "HighPrecisionTurtleSerializer")
register("n3-hp", Serializer, "src.serializer", "HighPrecisionN3Serializer")
register("trig-hp", Serializer, "src.serializer", "HighPrecisionTrigSerializer")

# Configure logging to stderr with INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

MODEL = Namespace("https://w3id.org/codata/model/")
CONCEPT = Namespace("https://w3id.org/codata/concepts/")
CONSTANT = Namespace("https://w3id.org/codata/constants/")
QUANTITY = Namespace("https://w3id.org/codata/quantities/")
UNIT = Namespace("https://w3id.org/codata/units/")
VERSION = Namespace("https://w3id.org/codata/constants/versions/")

DCTERMS = Namespace("http://purl.org/dc/terms/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/vocab/quantitykind/")
UCUM = Namespace("https://w3id.org/uom/")
SICONSTANT = Namespace("https://si-digital-framework.org/constants/")
SIUNIT = Namespace("https://si-digital-framework.org/SI/units/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
WIKIDATA = Namespace("https://www.wikidata.org/entity/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

JSONLD_CONTEXT = {
    "codata": str(MODEL),
    "concept": str(CONCEPT),
    "constant": str(CONSTANT),
    "quantity": str(QUANTITY),
    "unit": str(UNIT),
    "version": str(VERSION),
    "dcterms": str(DCTERMS),
    "owl": str(OWL),
    "qudt": str(QUDT),
    "schema": str(SCHEMA),
    "si-constant": str(SICONSTANT),
    "si-unit": str(SIUNIT),
    "skos": str(SKOS),
    "ucum": str(UCUM),
    "wikidata": str(WIKIDATA),
    "xsd": str(XSD),
    "uri": "@id",
    "id": "schema:identifier",
    "name": "skos:prefLabel",
    "nameFr": "skos:prefLabel",
    "definition": "skos:definition",
    "symbolTex": "codata:hasSymbol",
    "formula": "codata:formula",
    "formulaLatex": "codata:formulaLatex",
    "derivedFrom": "codata:derivedFrom",
    "value": "codata:value",
    "nistValue": "codata:nistValue",
    "valueSource": "codata:valueSource",
    "adjustmentRole": "codata:adjustmentRole",
    "isComputed": "codata:isComputed",
    "valueDecimal": "codata:valueDecimal",
    "valueFloat": "codata:valueFloat",
    "uncertainty": "codata:uncertainty",
    "uncertaintyDecimal": "codata:uncertaintyDecimal",
    "uncertaintyFloat": "codata:uncertaintyFloat",
    "nistExponent": "codata:nistExponent",
    "isExact": "codata:isExact",
    "isTruncated": "codata:isTruncated",
    "isNistTruncated": "codata:isNistTruncated",
    "significantDigits": "codata:significantDigits",
    "fitsFloat32": "codata:fitsFloat32",
    "fitsFloat64": "codata:fitsFloat64",
    "recommendedNumericType": "codata:recommendedNumericType",
    "isRatio": "codata:isRatio",
    "isRelationship": "codata:isRelationship",
    "exactMatch": "skos:exactMatch",
    "isDeprecated": "owl:deprecated",
    "versionId": "codata:versionId",
    "ucumCode": "ucum:ucum",
    "siExpression": "ucum:si_expression",
}


def with_jsonld_context(data: Any) -> Any:
    """Wrap a dict or list of dicts with the standard JSON-LD @context."""
    if isinstance(data, dict):
        return {"@context": JSONLD_CONTEXT, **data}
    if isinstance(data, list):
        return {"@context": JSONLD_CONTEXT, "@graph": data}
    return data


# Global RDF graph - loaded at module level before gunicorn forks workers
graph: Optional[Graph] = None

# Cache for SPARQL queries against the read-only RDF graph
QUERY_CACHE: dict[str, Any] = {}

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
    """Application lifespan: no-op as graph is loaded at module level."""
    yield


ROOT_PATH = os.getenv("ROOT_PATH", "")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=__version__,
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


@app.middleware("http")
async def handle_trailing_slashes(request: Request, call_next):
    path = request.url.path
    if path != "/" and path.endswith("/"):
        new_path = path.rstrip("/")
        root_path = request.scope.get("root_path", "")
        full_path = f"{root_path}{new_path}" if root_path else new_path
        query_string = request.url.query
        redirect_url = f"{full_path}?{query_string}" if query_string else full_path
        return RedirectResponse(url=redirect_url, status_code=307)
    return await call_next(request)


@app.middleware("http")
async def add_root_path_from_forwarded_prefix(request: Request, call_next):
    prefix = request.headers.get("x-forwarded-prefix")
    if prefix:
        prefix = prefix.rstrip("/")
        if prefix:
            request.scope["root_path"] = prefix
    return await call_next(request)


# Configure Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def _format_compact_number(val: str | None) -> str:
    if not val:
        return ""
    s = str(val).strip()
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$", s)
    if match:
        mantissa, exp = match.groups()
        if exp.startswith("+"):
            exp = exp[1:]
        max_m = max(8, 20 - len(exp) - 2)
        if len(mantissa) > max_m:
            mantissa = mantissa[:max_m] + "…"
        return f"{mantissa}e{exp}"
    return s[:24] + "…" if len(s) > 24 else s


templates.env.filters["compact_number"] = _format_compact_number


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTPException (includes 404 from routing)."""
    if exc.status_code == 404:
        path = request.url.path
        if path.endswith("/") and path != "/":
            new_path = path.rstrip("/")
            new_url = request.url.replace(path=new_path)
            return RedirectResponse(url=str(new_url), status_code=307)

        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Not Found: {request.url.path}",
                "path": request.url.path,
                "method": request.method,
            },
        )

    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException with path information for 404 errors."""
    response_content: dict[str, Any] = {"detail": exc.detail}
    if exc.status_code == 404:
        response_content["path"] = request.url.path
        response_content["method"] = request.method
    return JSONResponse(status_code=exc.status_code, content=response_content)


class GraphNotInitializedError(Exception):
    """Raised when the RDF graph is not initialized."""


class SparqlQueryError(Exception):
    """Raised when a SPARQL query fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


def _bind_standard_namespaces(g: Graph) -> None:
    g.bind("codata", MODEL)
    g.bind("concept", CONCEPT)
    g.bind("constant", CONSTANT)
    g.bind("dcterms", DCTERMS)
    g.bind("owl", OWL)
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


def load_rdf_data() -> Graph:
    """Load RDF graph from files in the data directory."""
    g = Graph()
    _bind_standard_namespaces(g)

    data_dir = settings.DATA_DIR
    if data_dir.exists():
        for f in sorted(data_dir.rglob("*")):
            fmt = _ext_to_format.get(f.suffix.lower())
            if fmt:
                try:
                    g.parse(str(f), format=fmt)
                except Exception as e:
                    logger.warning(f"Failed to parse {f} ({fmt}): {e}")
    else:
        logger.info(f"Data directory not found at {data_dir}")

    return g


logger.info("Loading RDF graph at module initialization...")
graph = load_rdf_data()
logger.info(f"Loaded {len(graph)} triples from RDF data files - shared across all workers")


def _require_graph() -> Graph:
    if graph is None:
        raise GraphNotInitializedError("RDF graph not initialized")
    return graph


def _get_label(g: Graph, uri: URIRef, lang: str = "en") -> Optional[str]:
    fallback = None
    for lit in g.objects(uri, SKOS.prefLabel):
        if isinstance(lit, Literal):
            if lit.language == lang:
                return str(lit)
            if lang == "en" and not lit.language:
                fallback = str(lit)
    return fallback


def _get_definition(g: Graph, uri: URIRef, lang: str = "en") -> Optional[str]:
    fallback = None
    for lit in g.objects(uri, SKOS.definition):
        if isinstance(lit, Literal):
            if lit.language == lang:
                return str(lit)
            if lang == "en" and not lit.language:
                fallback = str(lit)
    return fallback


def _get_bool(g: Graph, uri: URIRef, pred: URIRef) -> Optional[bool]:
    val = g.value(uri, pred)
    if isinstance(val, Literal):
        py_val = val.toPython()
        if isinstance(py_val, bool):
            return py_val
        return str(val).lower() == "true"
    return None


def _get_int(g: Graph, uri: URIRef, pred: URIRef) -> Optional[int]:
    val = g.value(uri, pred)
    if val is not None:
        try:
            return int(str(val))
        except (ValueError, TypeError):
            return None
    return None


def _get_str(g: Graph, uri: URIRef, pred: URIRef) -> Optional[str]:
    val = g.value(uri, pred)
    return str(val) if val is not None else None


def _extract_identifiers(g: Graph, uri: URIRef) -> Optional[list[Identifier]]:
    identifiers: list[Identifier] = []
    for ident_node in g.objects(uri, SCHEMA.identifier):
        if isinstance(ident_node, URIRef):
            prop_id = _get_str(g, ident_node, SCHEMA.propertyID) or str(ident_node).split("#")[-1]
            val = _get_str(g, ident_node, SCHEMA.value) or _get_str(g, ident_node, SCHEMA.identifier)
            url = _get_str(g, ident_node, SCHEMA.url)
            identifiers.append(Identifier(id=prop_id, value=val, url=url))
    if not identifiers:
        return None
    identifiers.sort(key=lambda x: x.id)
    return identifiers


def _extract_unit_summary(g: Graph, unit_uri: URIRef) -> Unit:
    u_id = str(unit_uri).split("/")[-1]
    pref_label = _get_label(g, unit_uri, "en")
    symbol_lit = None
    for ident in g.objects(unit_uri, SCHEMA.identifier):
        if isinstance(ident, Literal):
            symbol_lit = str(ident)
            break
    ucum_code = _get_str(g, unit_uri, UCUM.ucum)
    si_expr = _get_str(g, unit_uri, UCUM.si_expression)
    return Unit(
        id=u_id,
        uri=str(unit_uri),
        name=pref_label or symbol_lit or u_id,
        ucumCode=ucum_code,
        siExpression=si_expr,
    )


def _extract_constant_value(
    g: Graph,
    val_uri: URIRef,
    *,
    include_version_obj: bool = False,
) -> ConstantValue:
    val_str = _get_str(g, val_uri, MODEL.value) or ""
    unc_str = _get_str(g, val_uri, MODEL.uncertainty)
    val_dec_str = _get_str(g, val_uri, MODEL.valueDecimal)
    val_flt_str = _get_str(g, val_uri, MODEL.valueFloat)
    unc_dec_str = _get_str(g, val_uri, MODEL.uncertaintyDecimal)
    unc_flt_str = _get_str(g, val_uri, MODEL.uncertaintyFloat)
    version_uri = g.value(val_uri, MODEL.hasVersion)
    version_id = _get_str(g, val_uri, MODEL.versionId) or (
        str(version_uri).split("/")[-1] if version_uri else "unknown"
    )
    val_id = str(val_uri).split("/")[-1]

    version_obj = None
    if include_version_obj and isinstance(version_uri, URIRef):
        v_id = str(version_uri).split("/")[-1]
        version_obj = Version(id=v_id, uri=str(version_uri))

    return ConstantValue(
        id=val_id,
        uri=str(val_uri),
        versionId=version_id,
        value=val_str,
        nistValue=_get_str(g, val_uri, MODEL.nistValue),
        valueSource=_get_str(g, val_uri, MODEL.valueSource),
        adjustmentRole=_get_str(g, val_uri, MODEL.adjustmentRole),
        isComputed=_get_bool(g, val_uri, MODEL.isComputed),
        valueDecimal=Decimal(val_dec_str) if val_dec_str is not None else None,
        valueFloat=float(val_flt_str) if val_flt_str is not None else None,
        uncertainty=unc_str,
        uncertaintyDecimal=Decimal(unc_dec_str) if unc_dec_str is not None else None,
        uncertaintyFloat=float(unc_flt_str) if unc_flt_str is not None else None,
        nistExponent=_get_int(g, val_uri, MODEL.nistExponent),
        isExact=_get_bool(g, val_uri, MODEL.isExact),
        isTruncated=_get_bool(g, val_uri, MODEL.isTruncated),
        isNistTruncated=_get_bool(g, val_uri, MODEL.isNistTruncated),
        significantDigits=_get_int(g, val_uri, MODEL.significantDigits),
        fitsFloat32=_get_bool(g, val_uri, MODEL.fitsFloat32),
        fitsFloat64=_get_bool(g, val_uri, MODEL.fitsFloat64),
        recommendedNumericType=_get_str(g, val_uri, MODEL.recommendedNumericType),
        version=version_obj,
    )


def _resolve_constant_uri(g: Graph, id_or_alias: str) -> Optional[URIRef]:
    direct = CONSTANT[id_or_alias]
    if (direct, RDF.type, MODEL.Constant) in g:
        return direct
    target_low = id_or_alias.lower()
    # Check case-insensitive canonical ID or external identifier value (e.g. NIST ID like 'h', 'c', 'hbar')
    for c_uri in g.subjects(RDF.type, MODEL.Constant):
        if isinstance(c_uri, URIRef):
            c_id = str(c_uri).split("/")[-1]
            if c_id.lower() == target_low:
                return c_uri
            for ident_node in g.objects(c_uri, SCHEMA.identifier):
                if isinstance(ident_node, URIRef):
                    val = _get_str(g, ident_node, SCHEMA.value)
                    if val and val.lower() == target_low:
                        return c_uri
    return None


def _extract_constant(
    g: Graph,
    c_uri: URIRef,
    *,
    include_details: bool = False,
    version_filter: Optional[str] = None,
) -> Optional[Constant]:
    c_id = str(c_uri).split("/")[-1]
    c_name = _get_label(g, c_uri, "en") or c_id
    c_name_fr = _get_label(g, c_uri, "fr")
    is_deprecated = bool(_get_bool(g, c_uri, OWL.deprecated))

    # Quantity
    quantity = None
    q_uri = g.value(c_uri, MODEL.hasQuantity)
    if isinstance(q_uri, URIRef):
        q_id = str(q_uri).split("/")[-1]
        q_label = _get_label(g, q_uri, "en") or q_id
        quantity = Quantity(id=q_id, uri=str(q_uri), name=q_label)

    # Unit
    unit = None
    u_uri = g.value(c_uri, MODEL.hasUnit)
    if isinstance(u_uri, URIRef):
        unit = _extract_unit_summary(g, u_uri)

    # Values
    val_objs: list[ConstantValue] = []
    for val_uri in g.objects(c_uri, MODEL.hasValue):
        if isinstance(val_uri, URIRef):
            v_id = _get_str(g, val_uri, MODEL.versionId)
            if version_filter and v_id != version_filter:
                continue
            val_objs.append(_extract_constant_value(g, val_uri, include_version_obj=include_details))

    val_objs.sort(key=lambda v: v.versionId, reverse=True)
    if version_filter and not val_objs:
        return None

    current_val = val_objs[0] if val_objs else None
    hist_vals = val_objs[1:] if (include_details and len(val_objs) > 1) else None

    # Algebraic & formula relationships
    symbol_tex = _get_str(g, c_uri, MODEL.hasSymbol)
    formula = _get_str(g, c_uri, MODEL.formula)
    formula_latex = _get_str(g, c_uri, MODEL.formulaLatex)
    ucum_code = _get_str(g, c_uri, UCUM.ucum)

    derived_from: Optional[list[Constant]] = None
    deps = []
    for dep_uri in g.objects(c_uri, MODEL.derivedFrom):
        if isinstance(dep_uri, URIRef):
            dep_id = str(dep_uri).split("/")[-1]
            dep_name = _get_label(g, dep_uri, "en") or dep_id
            dep_sym = _get_str(g, dep_uri, MODEL.hasSymbol)
            deps.append(Constant(id=dep_id, uri=str(dep_uri), name=dep_name, symbolTex=dep_sym))
    if deps:
        deps.sort(key=lambda x: x.id)
        derived_from = deps

    used_in: Optional[list[Constant]] = None
    identifiers: Optional[list[Identifier]] = None
    if include_details:
        dependents = []
        for rev_uri in g.subjects(MODEL.derivedFrom, c_uri):
            if isinstance(rev_uri, URIRef):
                rev_id = str(rev_uri).split("/")[-1]
                rev_name = _get_label(g, rev_uri, "en") or rev_id
                rev_sym = _get_str(g, rev_uri, MODEL.hasSymbol)
                dependents.append(Constant(id=rev_id, uri=str(rev_uri), name=rev_name, symbolTex=rev_sym))
        if dependents:
            dependents.sort(key=lambda x: x.id)
            used_in = dependents
        identifiers = _extract_identifiers(g, c_uri)

    return Constant(
        id=c_id,
        uri=str(c_uri),
        name=c_name,
        nameFr=c_name_fr,
        symbolTex=symbol_tex,
        formula=formula,
        formulaLatex=formula_latex,
        derivedFrom=derived_from,
        usedIn=used_in,
        ucumCode=ucum_code,
        identifiers=identifiers,
        isDeprecated=True if is_deprecated else None,
        quantity=quantity,
        unit=unit,
        value=current_val,
        historicalValues=hist_vals,
    )


def format_sparql_results(results: Any, accept: str = "") -> Response:
    """Format SPARQL query results based on Accept header."""
    g = _require_graph()
    if getattr(results, "type", None) in ("SELECT", "ASK"):
        if "application/sparql-results+xml" in accept:
            return Response(
                results.serialize(format="xml"),
                media_type="application/sparql-results+xml",
            )
        return Response(
            results.serialize(format="json"),
            media_type="application/sparql-results+json",
        )
    else:
        result_graph = Graph()
        for prefix, namespace in g.namespace_manager.namespaces():
            result_graph.bind(prefix, namespace)

        for triple in results:
            result_graph.add(triple)

        return _serialize_rdf_graph(result_graph, accept)


def _serialize_rdf_graph(result_graph: Graph, accept: str) -> Response:
    if "application/rdf+xml" in accept or "application/xml" in accept:
        return Response(result_graph.serialize(format="xml"), media_type="application/rdf+xml")
    if "application/n-triples" in accept:
        return Response(result_graph.serialize(format="nt"), media_type="application/n-triples")
    if "text/turtle" in accept:
        return Response(result_graph.serialize(format="turtle-hp"), media_type="text/turtle")
    if "application/n3" in accept or "text/n3" in accept:
        return Response(result_graph.serialize(format="n3-hp"), media_type="text/n3")
    if "application/ld+json" in accept:
        return Response(
            result_graph.serialize(format="json-ld", context=JSONLD_CONTEXT),
            media_type="application/ld+json",
        )
    if "application/trig" in accept:
        return Response(result_graph.serialize(format="trig-hp"), media_type="application/trig")
    return Response(result_graph.serialize(format="turtle-hp"), media_type="text/turtle")


def run_sparql_query(query: str) -> Any:
    """Execute a SPARQL query against the global RDF graph. Returns raw results."""
    g = _require_graph()

    normalized_query = " ".join(query.split())
    query_key = f"{id(g)}:{hashlib.sha256(normalized_query.encode()).hexdigest()}"

    if query_key in QUERY_CACHE:
        return QUERY_CACHE[query_key]

    try:
        init_ns = dict(g.namespaces())
        prepared = prepareQuery(query, initNs=init_ns)
    except Exception as e:
        raise SparqlQueryError(f"SPARQL query error: {str(e)}", e)

    if _contains_service(prepared.algebra):
        raise SparqlQueryError("SPARQL SERVICE (federated query) is not supported by this endpoint")

    try:
        start_time = time.time()
        results = g.query(prepared)
        if results.type in ("SELECT", "ASK"):
            _ = results.bindings
        else:
            _ = results.graph
        duration = time.time() - start_time

        if len(QUERY_CACHE) < 1024:
            QUERY_CACHE[query_key] = results
        if duration > 0.5:
            logger.info(f"SPARQL query took {duration:.2f}s, cached (hash: {query_key[:16]}...)")

        return results
    except Exception as e:
        raise SparqlQueryError(f"SPARQL query error: {str(e)}", e)


def _contains_service(node: Any) -> bool:
    """True if the SPARQL algebra tree contains a SERVICE clause."""
    if isinstance(node, CompValue):
        if node.name == "ServiceGraphPattern":
            return True
        return any(_contains_service(v) for v in node.values())
    if isinstance(node, dict):
        return any(_contains_service(v) for v in node.values())
    if isinstance(node, (list, tuple, set)):
        return any(_contains_service(v) for v in node)
    return False


def build_sparql_query(query: str) -> str:
    """Prepopulate a SPARQL query with all defined namespace prefixes."""
    g = _require_graph()
    prefixes = ""
    for p, n in g.namespace_manager.namespaces():
        if p:
            prefixes += f"PREFIX {p}: <{n}>\n"
    return prefixes + "\n" + query.strip()


def _resolve_accept(request: Request) -> str:
    format_param = request.query_params.get("format", "").lower()
    format_map = {
        "html": "text/html",
        "json": "application/json",
        "jsonld": "application/ld+json",
        "turtle": "text/turtle",
        "ttl": "text/turtle",
        "rdfxml": "application/rdf+xml",
        "xml": "application/rdf+xml",
        "ntriples": "application/n-triples",
        "nt": "application/n-triples",
        "n3": "application/n3",
        "trig": "application/trig",
    }
    if format_param and format_param in format_map:
        return format_map[format_param]
    return request.headers.get("accept", "application/json")


def negotiate_content(
    uri: str,
    request: Request,
    json_response: JSONResponse,
    resource_data: Optional[dict[str, Any]] = None,
    template_name: Optional[str] = None,
) -> Response:
    """Handle content negotiation for a resource URI."""
    g = _require_graph()
    accept = _resolve_accept(request)

    if resource_data and template_name and ("text/html" in accept or "application/xhtml+xml" in accept):
        json_content = json_response.body.decode("utf-8")
        json_data = json.loads(json_content)
        json_ld = json.dumps(with_jsonld_context(json_data), indent=2)

        return templates.TemplateResponse(
            request,
            template_name,
            {
                "resource": resource_data,
                "json_ld": json_ld,
            },
        )

    if any(
        fmt in accept
        for fmt in [
            "text/turtle",
            "application/rdf+xml",
            "application/n-triples",
            "application/n3",
            "text/n3",
            "application/ld+json",
            "application/trig",
        ]
    ):
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

        for prefix, namespace in g.namespace_manager.namespaces():
            result_graph.bind(prefix, namespace)

        return _serialize_rdf_graph(result_graph, accept)

    return json_response


def _get_dataset_stats() -> dict[str, Any]:
    g = _require_graph()
    all_consts = list(g.subjects(RDF.type, MODEL.Constant))
    deprecated_consts = [c for c in all_consts if isinstance(c, URIRef) and _get_bool(g, c, OWL.deprecated)]
    active_consts = len(all_consts) - len(deprecated_consts)
    all_vals = list(g.subjects(RDF.type, MODEL.ConstantValue))
    computed_upgrades = [
        v for v in all_vals if isinstance(v, URIRef) and _get_str(g, v, MODEL.valueSource) == "computed"
    ]
    quantities_cnt = len(list(g.subjects(RDF.type, MODEL.Quantity)))
    concepts_cnt = len(list(g.subjects(RDF.type, MODEL.Concept)))
    units_cnt = len(list(g.subjects(RDF.type, MODEL.Unit)))
    versions_list = sorted(
        [
            _get_str(g, v, SCHEMA.identifier) or str(v).split("/")[-1]
            for v in g.subjects(RDF.type, MODEL.Version)
            if isinstance(v, URIRef)
        ],
        reverse=True,
    )
    ont_uri = URIRef("https://w3id.org/codata/model/")
    ont_version = _get_str(g, ont_uri, OWL.versionInfo) or "0.4.0"

    return {
        "apiVersion": __version__,
        "ontologyVersion": ont_version,
        "triples": len(g),
        "constantsTotal": len(all_consts),
        "constantsActive": active_consts,
        "constantsDeprecated": len(deprecated_consts),
        "constantValuesTotal": len(all_vals),
        "computedUpgradesTotal": len(computed_upgrades),
        "quantities": quantities_cnt,
        "concepts": concepts_cnt,
        "units": units_cnt,
        "versions": versions_list,
        "latestVersion": versions_list[0] if versions_list else "2022",
    }


@app.get("/")
async def index(request: Request):
    """API Landing page & Developer Portal with HTML and JSON support."""
    accept = _resolve_accept(request)

    if "text/html" not in accept and "application/xhtml+xml" not in accept:
        return {
            "message": "Welcome to CODATA DRUM Physical Fundamental Constants API!",
            "version": __version__,
        }

    stats = _get_dataset_stats()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "message": "Welcome to CODATA DRUM Physical Fundamental Constants API!",
            "stats": stats,
        },
    )


@app.get("/health")
async def health():
    """Health and dataset observability endpoint."""
    try:
        stats = _get_dataset_stats()
        return {"status": "healthy", **stats}
    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")


@app.get("/ontology")
async def ontology(request: Request):
    """Serve the formal CODATA OWL/RDFS ontology with content negotiation."""
    ont_path = settings.DATA_DIR / "codata.ttl"
    if not ont_path.exists():
        raise HTTPException(status_code=404, detail="Ontology file not found")
    ont_graph = Graph()
    _bind_standard_namespaces(ont_graph)
    ont_graph.parse(str(ont_path), format="turtle")
    accept = _resolve_accept(request)
    if "application/json" in accept and "ld+json" not in accept:
        return Response(
            ont_graph.serialize(format="json-ld", context=JSONLD_CONTEXT),
            media_type="application/ld+json",
        )
    return _serialize_rdf_graph(ont_graph, accept)


@app.get("/explorer", response_class=HTMLResponse)
async def explorer(request: Request):
    """Serve the Read-Only Interactive Knowledge Explorer for scientists and non-technical users."""
    stats = _get_dataset_stats()
    return templates.TemplateResponse(
        request,
        "explorer.html",
        {"stats": stats},
    )


@app.get("/concepts", response_model=List[Concept], response_model_exclude_none=True)
async def concepts(request: Request):
    """Retrieve all concepts (including quantities) and return as structured data."""
    try:
        g = _require_graph()
        concepts_list = []
        for c_uri in g.subjects(RDF.type, MODEL.Concept):
            if not isinstance(c_uri, URIRef):
                continue
            c_id = str(c_uri).split("/")[-1]
            c_label = _get_label(g, c_uri, "en") or c_id
            c_label_fr = _get_label(g, c_uri, "fr")
            c_def = _get_definition(g, c_uri, "en")
            is_quantity = (c_uri, RDF.type, MODEL.Quantity) in g
            is_ratio = _get_bool(g, c_uri, MODEL.isRatio)
            is_rel = _get_bool(g, c_uri, MODEL.isRelationship)
            exact_matches = [str(m) for m in g.objects(c_uri, SKOS.exactMatch)]
            concept_obj = Concept(
                id=c_id,
                uri=str(c_uri),
                name=c_label,
                nameFr=c_label_fr,
                definition=c_def,
                isQuantity=is_quantity,
                isRatio=is_ratio,
                isRelationship=is_rel,
                exactMatch=exact_matches if exact_matches else None,
            )
            concepts_list.append(concept_obj.model_dump(exclude_none=True))

        concepts_list.sort(key=lambda x: x["name"].lower())

        accept = _resolve_accept(request)
        if "text/html" in accept:
            json_ld = json.dumps(with_jsonld_context(concepts_list), indent=2)
            return templates.TemplateResponse(
                request,
                "concepts.html",
                {"resources": concepts_list, "json_ld": json_ld},
            )
        if "application/ld+json" in accept:
            return JSONResponse(content=with_jsonld_context(concepts_list), media_type="application/ld+json")

        return JSONResponse(content=concepts_list)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/concepts/{id}", response_model=Concept, response_model_exclude_none=True)
async def concept(id: str, request: Request):
    """Retrieve concept information with content negotiation support."""
    try:
        g = _require_graph()
        concept_uri = CONCEPT[id]
        quantity_uri = QUANTITY[id]

        target_uri: Optional[URIRef] = None
        if (concept_uri, RDF.type, MODEL.Concept) in g:
            target_uri = concept_uri
        elif (quantity_uri, RDF.type, MODEL.Concept) in g:
            target_uri = quantity_uri

        if target_uri is None:
            raise HTTPException(status_code=404, detail=f"Concept '{id}' not found")

        label = _get_label(g, target_uri, "en") or id
        label_fr = _get_label(g, target_uri, "fr")
        definition = _get_definition(g, target_uri, "en")
        is_quantity = (target_uri, RDF.type, MODEL.Quantity) in g
        is_ratio = _get_bool(g, target_uri, MODEL.isRatio)
        is_rel = _get_bool(g, target_uri, MODEL.isRelationship)
        exact_matches = [str(m) for m in g.objects(target_uri, SKOS.exactMatch)]

        parts = []
        for p_uri in g.objects(target_uri, DCTERMS.hasPart):
            if isinstance(p_uri, URIRef):
                p_id = str(p_uri).split("/")[-1]
                p_name = _get_label(g, p_uri, "en") or p_id
                parts.append(Concept(id=p_id, uri=str(p_uri), name=p_name))
        parts.sort(key=lambda x: x.name.lower())

        broader_concepts = []
        for b_uri in g.objects(target_uri, SKOS.broader):
            if isinstance(b_uri, URIRef):
                b_id = str(b_uri).split("/")[-1]
                b_name = _get_label(g, b_uri, "en") or b_id
                broader_concepts.append(Concept(id=b_id, uri=str(b_uri), name=b_name))
        broader_concepts.sort(key=lambda x: x.name.lower())

        narrower_concepts = []
        for n_uri in g.subjects(SKOS.broader, target_uri):
            if isinstance(n_uri, URIRef):
                n_id = str(n_uri).split("/")[-1]
                n_name = _get_label(g, n_uri, "en") or n_id
                narrower_concepts.append(Concept(id=n_id, uri=str(n_uri), name=n_name))
        narrower_concepts.sort(key=lambda x: x.name.lower())

        concept_quantity = None
        concept_quantities = None
        if is_quantity:
            concept_quantity = Quantity(id=id, uri=str(target_uri), name=label)
        else:
            q_list = []
            for q_uri in g.subjects(DCTERMS.hasPart, target_uri):
                if isinstance(q_uri, URIRef) and (q_uri, RDF.type, MODEL.Quantity) in g:
                    q_id = str(q_uri).split("/")[-1]
                    q_name = _get_label(g, q_uri, "en") or q_id
                    q_list.append(Quantity(id=q_id, uri=str(q_uri), name=q_name))
            if q_list:
                q_list.sort(key=lambda x: x.name.lower())
                concept_quantities = q_list

        concept_obj = Concept(
            id=id,
            uri=str(target_uri),
            name=label,
            nameFr=label_fr,
            definition=definition,
            isQuantity=is_quantity,
            isRatio=is_ratio,
            isRelationship=is_rel,
            exactMatch=exact_matches if exact_matches else None,
            parts=parts if parts else None,
            broader=broader_concepts if broader_concepts else None,
            narrower=narrower_concepts if narrower_concepts else None,
            quantity=concept_quantity,
            quantities=concept_quantities,
        )

        resource_data = concept_obj.model_dump(exclude_none=True)
        json_response = JSONResponse(content=resource_data)
        return negotiate_content(str(target_uri), request, json_response, resource_data, "concept.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/quantities", response_model=List[Quantity], response_model_exclude_none=True)
async def quantities(request: Request):
    """Retrieve all quantities and return as structured data."""
    try:
        g = _require_graph()
        quantities_list: list[Quantity] = []
        for q_uri in g.subjects(RDF.type, MODEL.Quantity):
            if not isinstance(q_uri, URIRef):
                continue
            q_id = str(q_uri).split("/")[-1]
            q_label = _get_label(g, q_uri, "en") or q_id
            q_def = _get_definition(g, q_uri, "en")
            is_ratio = _get_bool(g, q_uri, MODEL.isRatio)
            is_rel = _get_bool(g, q_uri, MODEL.isRelationship)

            q_consts = []
            for c_uri in g.objects(q_uri, MODEL.hasConstant):
                if isinstance(c_uri, URIRef):
                    c_id = str(c_uri).split("/")[-1]
                    c_label = _get_label(g, c_uri, "en") or c_id
                    q_consts.append(Constant(id=c_id, uri=str(c_uri), name=c_label))
            q_consts.sort(key=lambda x: x.name.lower())

            quantities_list.append(
                Quantity(
                    id=q_id,
                    uri=str(q_uri),
                    name=q_label,
                    definition=q_def,
                    isRatio=is_ratio,
                    isRelationship=is_rel,
                    constants=q_consts if q_consts else None,
                )
            )

        quantities_list.sort(key=lambda x: x.name.lower())
        quantities_json = [q.model_dump(exclude_none=True) for q in quantities_list]

        accept = _resolve_accept(request)
        if "text/html" in accept:
            json_ld = json.dumps(with_jsonld_context(quantities_json), indent=2)
            return templates.TemplateResponse(
                request,
                "quantities.html",
                {"resources": quantities_json, "json_ld": json_ld},
            )
        if "application/ld+json" in accept:
            return JSONResponse(content=with_jsonld_context(quantities_json), media_type="application/ld+json")

        return JSONResponse(content=quantities_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/quantities/{id}", response_model=Quantity, response_model_exclude_none=True)
async def quantity(id: str, request: Request):
    """Retrieve quantity information with content negotiation support."""
    try:
        g = _require_graph()
        uri = QUANTITY[id]
        if (uri, RDF.type, MODEL.Quantity) not in g:
            raise HTTPException(status_code=404, detail=f"Quantity '{id}' not found")

        label = _get_label(g, uri, "en") or id
        definition = _get_definition(g, uri, "en")
        is_ratio = _get_bool(g, uri, MODEL.isRatio)
        is_rel = _get_bool(g, uri, MODEL.isRelationship)

        q_concepts = []
        for c_uri in g.objects(uri, DCTERMS.hasPart):
            if isinstance(c_uri, URIRef):
                c_id = str(c_uri).split("/")[-1]
                c_name = _get_label(g, c_uri, "en") or c_id
                q_concepts.append(Concept(id=c_id, uri=str(c_uri), name=c_name))
        q_concepts.sort(key=lambda x: x.name.lower())

        q_constants = []
        for const_uri in g.objects(uri, MODEL.hasConstant):
            if isinstance(const_uri, URIRef):
                const_id = str(const_uri).split("/")[-1]
                const_name = _get_label(g, const_uri, "en") or const_id
                const_sym = _get_str(g, const_uri, MODEL.hasSymbol)
                q_constants.append(Constant(id=const_id, uri=str(const_uri), name=const_name, symbolTex=const_sym))
        q_constants.sort(key=lambda x: x.name.lower())

        quantity_obj = Quantity(
            id=id,
            uri=str(uri),
            name=label,
            definition=definition,
            isRatio=is_ratio,
            isRelationship=is_rel,
            concepts=q_concepts if q_concepts else None,
            constants=q_constants if q_constants else None,
        )

        resource_data = quantity_obj.model_dump(exclude_none=True)
        json_response = JSONResponse(content=resource_data)
        return negotiate_content(str(uri), request, json_response, resource_data, "quantity.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/units", response_model=List[Unit], response_model_exclude_none=True)
async def units(request: Request):
    """Retrieve all units and return as structured data."""
    try:
        g = _require_graph()
        units_list: list[Unit] = []
        for u_uri in g.subjects(RDF.type, MODEL.Unit):
            if isinstance(u_uri, URIRef):
                units_list.append(_extract_unit_summary(g, u_uri))

        units_list.sort(key=lambda x: x.id.lower())
        units_json = [u.model_dump(exclude_none=True) for u in units_list]

        accept = _resolve_accept(request)
        if "text/html" in accept:
            json_ld = json.dumps(with_jsonld_context(units_json), indent=2)
            return templates.TemplateResponse(
                request,
                "units.html",
                {"resources": units_json, "json_ld": json_ld},
            )
        if "application/ld+json" in accept:
            return JSONResponse(content=with_jsonld_context(units_json), media_type="application/ld+json")

        return JSONResponse(content=units_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants", response_model=List[Constant], response_model_exclude_none=True)
async def constants(
    request: Request,
    include_deprecated: bool = False,
    version: Optional[str] = None,
    q: Optional[str] = None,
    role: Optional[str] = None,
    is_exact: Optional[bool] = None,
    is_computed: Optional[bool] = None,
    value_source: Optional[str] = None,
    recommended_numeric_type: Optional[str] = None,
    quantity: Optional[str] = None,
    unit: Optional[str] = None,
):
    """Retrieve constants and return as structured data with rich filtering.

    By default, returns only active constants from the latest CODATA release (2022).
    """
    try:
        g = _require_graph()
        q_low = q.strip().lower() if q else None

        constants_objs: list[Constant] = []
        for c_uri in g.subjects(RDF.type, MODEL.Constant):
            if not isinstance(c_uri, URIRef):
                continue
            const_obj = _extract_constant(g, c_uri, include_details=False, version_filter=version)
            if const_obj is None:
                continue

            if not version and not include_deprecated and const_obj.isDeprecated:
                continue

            val = const_obj.value
            if role and (val is None or val.adjustmentRole != role):
                continue
            if is_exact is not None and (val is None or bool(val.isExact) != is_exact):
                continue
            if is_computed is not None and (val is None or bool(val.isComputed) != is_computed):
                continue
            if value_source and (val is None or val.valueSource != value_source):
                continue
            if recommended_numeric_type and (val is None or val.recommendedNumericType != recommended_numeric_type):
                continue
            if quantity and (const_obj.quantity is None or const_obj.quantity.id != quantity):
                continue
            if unit and (const_obj.unit is None or const_obj.unit.id != unit):
                continue

            if q_low:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            const_obj.id,
                            const_obj.name,
                            const_obj.nameFr,
                            const_obj.symbolTex,
                            const_obj.formula,
                        ],
                    )
                ).lower()
                if q_low not in haystack:
                    continue

            constants_objs.append(const_obj)

        constants_objs.sort(key=lambda c: c.name.lower())
        constants_list = [c.model_dump(exclude_none=True) for c in constants_objs]

        accept = _resolve_accept(request)
        if "text/html" in accept:
            json_ld = json.dumps(with_jsonld_context(constants_list), indent=2)
            return templates.TemplateResponse(
                request,
                "constants.html",
                {
                    "resources": constants_list,
                    "json_ld": json_ld,
                    "filters": {
                        "version": version or "2022",
                        "role": role or "",
                        "value_source": value_source or "",
                        "recommended_numeric_type": recommended_numeric_type or "",
                        "q": q or "",
                        "include_deprecated": include_deprecated,
                    },
                },
            )
        if "application/ld+json" in accept:
            return JSONResponse(content=with_jsonld_context(constants_list), media_type="application/ld+json")

        return JSONResponse(content=constants_list)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/versions", response_model=List[Version], response_model_exclude_none=True)
async def versions(request: Request):
    """Retrieve all CODATA release versions and return as structured data."""
    try:
        g = _require_graph()
        versions_list: list[Version] = []
        for v_uri in g.subjects(RDF.type, MODEL.Version):
            if not isinstance(v_uri, URIRef):
                continue
            v_id = _get_str(g, v_uri, SCHEMA.identifier) or str(v_uri).split("/")[-1]
            issued_str = _get_str(g, v_uri, DCTERMS.issued)
            published = date.fromisoformat(issued_str) if issued_str else None
            versions_list.append(Version(id=v_id, uri=str(v_uri), published=published))

        versions_list.sort(key=lambda x: x.id)
        versions_json = [v.model_dump(exclude_none=True) for v in versions_list]

        accept = _resolve_accept(request)
        if "text/html" in accept:
            json_ld = json.dumps(with_jsonld_context(versions_json), indent=2)
            return templates.TemplateResponse(
                request,
                "versions.html",
                {"resources": versions_json, "json_ld": json_ld},
            )
        if "application/ld+json" in accept:
            return JSONResponse(content=with_jsonld_context(versions_json), media_type="application/ld+json")

        return JSONResponse(content=versions_json)

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/versions/{id}", response_model=Version, response_model_exclude_none=True)
async def version(id: str, request: Request):
    """Retrieve version information with content negotiation support."""
    try:
        g = _require_graph()
        uri = VERSION[id]
        if (uri, RDF.type, MODEL.Version) not in g:
            raise HTTPException(status_code=404, detail=f"Version '{id}' not found")

        issued_str = _get_str(g, uri, DCTERMS.issued)
        published = date.fromisoformat(issued_str) if issued_str else None

        v_constants: list[Constant] = []
        seen_c: set[str] = set()
        for val_uri in g.subjects(MODEL.hasVersion, uri):
            if not isinstance(val_uri, URIRef):
                continue
            c_uri = g.value(val_uri, DCTERMS.isVersionOf)
            if isinstance(c_uri, URIRef) and str(c_uri) not in seen_c:
                seen_c.add(str(c_uri))
                c_id = str(c_uri).split("/")[-1]
                c_label = _get_label(g, c_uri, "en") or c_id
                v_constants.append(Constant(id=c_id, uri=str(c_uri), name=c_label))

        v_constants.sort(key=lambda x: x.name.lower())

        version_obj = Version(
            id=id,
            uri=str(uri),
            published=published,
            constants=v_constants if v_constants else None,
        )

        resource_data = version_obj.model_dump(exclude_none=True)
        json_response = JSONResponse(content=resource_data)
        return negotiate_content(str(uri), request, json_response, resource_data, "version.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/{id}", response_model=Constant, response_model_exclude_none=True)
async def constant(id: str, request: Request):
    """Retrieve constant information with content negotiation support.

    Supports lookup by canonical Constant ID (e.g. `PlanckConstant`) or external identifier
    such as NIST symbol (e.g. `h`, `c`, `hbar`).
    """
    try:
        g = _require_graph()
        c_uri = _resolve_constant_uri(g, id)
        if c_uri is None:
            raise HTTPException(status_code=404, detail=f"Constant '{id}' not found")

        constant_obj = _extract_constant(g, c_uri, include_details=True)
        if constant_obj is None:
            raise HTTPException(status_code=404, detail=f"Constant '{id}' not found")

        resource_data = constant_obj.model_dump(exclude_none=True)
        json_response = JSONResponse(content=resource_data)
        return negotiate_content(str(c_uri), request, json_response, resource_data, "constant.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/constants/{id}/{versionId}", response_model=ConstantValue, response_model_exclude_none=True)
async def constant_value(id: str, versionId: str, request: Request):
    """Retrieve a specific ConstantValue for a constant in a given CODATA version."""
    try:
        g = _require_graph()
        c_uri = _resolve_constant_uri(g, id)
        if c_uri is None:
            raise HTTPException(
                status_code=404,
                detail=f"ConstantValue for constant '{id}' in version '{versionId}' not found",
            )

        canonical_id = str(c_uri).split("/")[-1]
        val_uri = CONSTANT[f"{canonical_id}/{versionId}"]
        if (val_uri, RDF.type, MODEL.ConstantValue) not in g:
            raise HTTPException(
                status_code=404,
                detail=f"ConstantValue for constant '{id}' in version '{versionId}' not found",
            )

        cv_obj = _extract_constant_value(g, val_uri, include_version_obj=True)
        const_obj = _extract_constant(g, c_uri, include_details=False, version_filter=versionId)
        if const_obj is not None:
            const_obj.value = cv_obj
            resource_data = const_obj.model_dump(exclude_none=True)
        else:
            resource_data = cv_obj.model_dump(exclude_none=True)

        json_response = JSONResponse(content=cv_obj.model_dump(exclude_none=True))
        return negotiate_content(str(val_uri), request, json_response, resource_data, "constant.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/units/{id}", response_model=Unit, response_model_exclude_none=True)
async def unit(id: str, request: Request):
    """Retrieve unit information with content negotiation support."""
    try:
        g = _require_graph()
        uri = UNIT[id]
        if (uri, RDF.type, MODEL.Unit) not in g:
            raise HTTPException(status_code=404, detail=f"Unit '{id}' not found")

        unit_obj = _extract_unit_summary(g, uri)
        unit_obj.identifiers = _extract_identifiers(g, uri)

        u_constants: list[Constant] = []
        for c_uri in g.subjects(MODEL.hasUnit, uri):
            if isinstance(c_uri, URIRef) and (c_uri, RDF.type, MODEL.Constant) in g:
                c_id = str(c_uri).split("/")[-1]
                c_label = _get_label(g, c_uri, "en") or c_id
                c_sym = _get_str(g, c_uri, MODEL.hasSymbol)
                u_constants.append(Constant(id=c_id, uri=str(c_uri), name=c_label, symbolTex=c_sym))
        u_constants.sort(key=lambda x: x.name.lower())
        unit_obj.constants = u_constants if u_constants else None

        resource_data = unit_obj.model_dump(exclude_none=True)
        json_response = JSONResponse(content=resource_data)
        return negotiate_content(str(uri), request, json_response, resource_data, "unit.html")

    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=f"SPARQL query error: {e.message}")


@app.get("/sparql")
async def sparql_get(request: Request, query: Optional[str] = None):
    """Execute a SPARQL query via GET request (SPARQL 1.1 Protocol §2.1.1),
    or render the interactive SPARQL Console when opened in a web browser without a query.
    """
    accept = _resolve_accept(request)
    if not query:
        if "text/html" in accept or "application/xhtml+xml" in accept:
            return templates.TemplateResponse(request, "sparql.html", {})
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")

    try:
        results = run_sparql_query(query)
    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=e.message)

    return format_sparql_results(results, accept)


@app.post("/sparql")
async def sparql_post(request: Request):
    """Execute a SPARQL query via POST request (SPARQL 1.1 Protocol §2.1.2 and §2.1.3)."""
    content_type = request.headers.get("content-type", "")

    if "application/x-www-form-urlencoded" in content_type:
        try:
            form_data = await request.form()
            query = str(form_data.get("query") or "")
        except (AssertionError, Exception):
            body = await request.body()
            params = urllib.parse.parse_qs(body.decode("utf-8"))
            query_vals = params.get("query")
            query = query_vals[0] if query_vals else ""
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' parameter")
    elif "application/sparql-query" in content_type:
        body = await request.body()
        query = body.decode("utf-8")
        if not query:
            raise HTTPException(status_code=400, detail="Empty query body")
    elif "application/json" in content_type:
        try:
            json_body = await request.json()
            query = json_body.get("query")
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query' field in JSON body")
        except HTTPException:
            raise
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

    accept = _resolve_accept(request)
    return format_sparql_results(results, accept)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
