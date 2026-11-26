import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import ctypes
from datetime import date
from decimal import Decimal
import hashlib
import gc
import json
import logging
import os
from pathlib import Path
import re
try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]
import select
import signal
import sys
import threading
import time
from typing import Any, List, Optional
import urllib.parse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.plugin import register
import rdflib.plugins.sparql
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import SPARQLError
from rdflib.serializer import Serializer
from pyparsing import ParseBaseException
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

class BoundedResultCache:
    """LRU cache for SPARQL results, bounded by entry count and by total result size.

    Size is measured in result rows (SELECT) or triples (CONSTRUCT/DESCRIBE).
    Row and triple counts are only an approximate indicator of memory use (a
    row can hold arbitrarily long literals): these limits constrain the number
    and cardinality of retained results, not memory in bytes. Results larger
    than ``max_entry_rows`` are never cached; ``max_entries == 0`` disables
    the cache.

    Not thread-safe: use it from one thread (the event loop) only.
    """

    def __init__(self, max_entries: int, max_total_rows: int, max_entry_rows: int):
        self.max_entries = max_entries
        self.max_total_rows = max_total_rows
        self.max_entry_rows = max_entry_rows
        self._data: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._total_rows = 0

    def get(self, key: str) -> Any:
        item = self._data.get(key)
        if item is None:
            return None
        self._data.move_to_end(key)
        return item[0]

    def put(self, key: str, value: Any, rows: int) -> None:
        if rows > min(self.max_entry_rows, self.max_total_rows) or self.max_entries <= 0:
            return
        if key in self._data:
            self._total_rows -= self._data.pop(key)[1]
        self._data[key] = (value, rows)
        self._total_rows += rows
        while self._data and (len(self._data) > self.max_entries or self._total_rows > self.max_total_rows):
            _, (_, evicted_rows) = self._data.popitem(last=False)
            self._total_rows -= evicted_rows

    def clear(self) -> None:
        self._data.clear()
        self._total_rows = 0

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def __len__(self) -> int:
        return len(self._data)


def _result_rows(results: Any) -> int:
    """Size of a SPARQL result: rows for SELECT, 1 for ASK, triples for CONSTRUCT/DESCRIBE."""
    if results.type == "SELECT":
        return len(results.bindings)
    if results.type == "ASK":
        return 1
    return len(results.graph)


# Cache for SPARQL queries against the read-only RDF graph
QUERY_CACHE = BoundedResultCache(
    max_entries=settings.SPARQL_CACHE_MAX_ENTRIES,
    max_total_rows=settings.SPARQL_CACHE_MAX_TOTAL_ROWS,
    max_entry_rows=settings.SPARQL_CACHE_MAX_ENTRY_ROWS,
)

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
    # Trailing slashes are handled by handle_trailing_slashes(). Starlette's own
    # redirect_slashes builds an absolute URL from the Host header; keep it off.
    redirect_slashes=False,
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
    # The fully decoded path. (request.url.path is partly re-encoded: "?" and
    # "#" are escaped, "%" is not, so it can be neither used nor quoted as is.)
    path = request.scope["path"]
    root = request.scope.get("root_path", "")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root):] or "/"  # some servers (uvicorn --root-path) include it
    if path != "/" and path.endswith("/") and not any(seg in (".", "..") for seg in path.split("/")):
        # Collapse leading slashes too: "//evil.example/" must not become the
        # protocol-relative redirect target "//evil.example". Paths with dot
        # segments are not redirected (they get a 404 from routing), so a
        # redirect can never leave the application's root path.
        #
        # The path is *decoded*; percent-encode it again before it goes into
        # the Location URL. Otherwise a request for "/%252e%252e/" (decoded:
        # "/%2e%2e/") would redirect to ".../%2e%2e", which browsers treat as
        # "..", and decoded "?", "#" or "%" would turn path data into URL syntax.
        new_path = urllib.parse.quote("/" + path.strip("/"), safe="/")
        root_path = request.scope.get("root_path", "")
        full_path = f"{root_path}{new_path}" if root_path else new_path
        query_string = request.url.query
        redirect_url = f"{full_path}?{query_string}" if query_string else full_path
        return RedirectResponse(url=redirect_url, status_code=307)
    return await call_next(request)


_PREFIX_SEGMENT = re.compile(r"[A-Za-z0-9._~-]+")


def _valid_forwarded_prefix(value: str) -> Optional[str]:
    """Return a safe root_path from an X-Forwarded-Prefix value, or None.

    The value ends up in redirect Location headers and in every link on the HTML
    pages, so only a plain absolute path is accepted: "/" followed by one or more
    segments of RFC 3986 unreserved characters, with at most one trailing slash
    (removed). Everything else is rejected, including "//host", schemes, empty,
    "." or ".." segments, backslashes and percent-encoding. If
    FORWARDED_PREFIX_ALLOWED is non-empty, the value must also be one of those.
    """
    prefix = value[:-1] if value.endswith("/") else value
    if not prefix.startswith("/"):
        return None
    segments = prefix[1:].split("/")
    if any(seg in (".", "..") or not _PREFIX_SEGMENT.fullmatch(seg) for seg in segments):
        return None
    allowed = settings.FORWARDED_PREFIX_ALLOWED
    if allowed and prefix not in allowed:
        return None
    return prefix


class _RateLimitedWarning:
    """Log a warning at most once per interval, with a count of suppressed ones."""

    def __init__(self, interval: float = 60.0):
        self.interval = interval
        self._last = float("-inf")
        self._suppressed = 0

    def __call__(self, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last < self.interval:
            self._suppressed += 1
            return
        if self._suppressed:
            msg += f" ({self._suppressed} similar messages suppressed)"
        self._last, self._suppressed = now, 0
        logger.warning(msg, *args)


_warn_bad_prefix = _RateLimitedWarning()


@app.middleware("http")
async def add_root_path_from_forwarded_prefix(request: Request, call_next):
    header = request.headers.get("x-forwarded-prefix")
    if header and settings.TRUST_X_FORWARDED_PREFIX:
        prefix = _valid_forwarded_prefix(header)
        if prefix:
            request.scope["root_path"] = prefix
        else:
            _warn_bad_prefix("Ignoring invalid or non-allowed X-Forwarded-Prefix: %r", header[:200])
    return await call_next(request)


# Configure Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

class SubpathStaticFiles(StaticFiles):
    """StaticFiles subclass that properly handles reverse proxies with stripped subpaths.

    When running behind a reverse proxy that sets X-Forwarded-Prefix or root_path
    and strips the prefix before forwarding to the backend, Starlette's StaticFiles
    mount can fail to strip the /static prefix from the relative path. This subclass
    normalizes the file path so static assets resolve seamlessly regardless of
    whether the proxy stripped the path or preserved the prefix.
    """

    def get_path(self, scope: Any) -> str:
        from starlette.staticfiles import get_route_path

        route_path = get_route_path(scope)
        for prefix in ["/static/", "static/", "/"]:
            if route_path.startswith(prefix):
                route_path = route_path[len(prefix) :]
                break
        return os.path.normpath(os.path.join(*route_path.split("/")))


# Configure static file serving
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    for candidate in [
        Path.cwd() / "src" / "static",
        Path.cwd() / "static",
        Path(__file__).resolve().parent.parent / "src" / "static",
    ]:
        if candidate.exists():
            static_dir = candidate
            break

if static_dir.exists():
    app.mount("/static", SubpathStaticFiles(directory=str(static_dir)), name="static")
    if ROOT_PATH and ROOT_PATH != "/":
        clean_root = ROOT_PATH.rstrip("/")
        app.mount(f"{clean_root}/static", SubpathStaticFiles(directory=str(static_dir)), name="static_subpath")
    logger.info("Mounted static files from %s at /static", static_dir)
else:
    logger.warning("Static directory not found; static assets will not be served")


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
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Not Found: {request.url.path}",
                "path": request.url.path,
                "method": request.method,
            },
        )

    return JSONResponse(
        status_code=exc.status_code, content={"detail": str(exc.detail)}, headers=getattr(exc, "headers", None)
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException with path information for 404 errors."""
    response_content: dict[str, Any] = {"detail": exc.detail}
    if exc.status_code == 404:
        response_content["path"] = request.url.path
        response_content["method"] = request.method
    return JSONResponse(status_code=exc.status_code, content=response_content, headers=exc.headers)


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


def _is_client_query_error(e: BaseException) -> bool:
    """Heuristic: True if an exception from parsing/evaluating a query is the query's fault.

    rdflib has no single exception type for mistakes in a query. Measured with
    rdflib 7.6: syntax errors raise pyparsing ParseException, an unknown prefix
    raises a plain Exception, a bad regex raises re.error, STRLANG with a
    non-string or an out-of-range LIMIT raise ValueError, deep nesting raises
    RecursionError, and evaluation errors can raise SPARQLError. Everything
    else (TypeError, AttributeError, KeyError, MemoryError, UnicodeError, ...)
    is treated as a server-side failure.

    This cannot be exact: an internal ValueError or plain Exception is also
    classified as a client error (and its message, bounded in length, returned
    with 400). Re-run tests/test_security.py when upgrading rdflib or pyparsing.
    """
    if isinstance(e, UnicodeError):
        return False
    if isinstance(e, (ParseBaseException, SPARQLError, RecursionError, re.error, ValueError)):
        return True
    return type(e) is Exception


def _query_error(e: Exception) -> "SparqlQueryError":
    if isinstance(e, RecursionError):
        return SparqlQueryError("SPARQL query is nested too deeply", e)
    return SparqlQueryError(f"SPARQL query error: {str(e)}", e)


def run_sparql_query(query: str, *, use_cache: bool = True) -> Any:
    """Execute a SPARQL query against the global RDF graph. Returns raw results.

    use_cache=False bypasses QUERY_CACHE completely (no lookup, no insertion);
    client-supplied queries on /sparql always use it. Exceptions that
    _is_client_query_error() classifies as mistakes in the query are raised as
    SparqlQueryError; all others propagate unchanged.
    """
    g = _require_graph()

    # Key on the exact query text. Collapsing whitespace is not safe: it also
    # changes string literals, so different queries ("a  b" vs "a b") would
    # share a cache entry and the second would get the first one's answer.
    query_key = f"{id(g)}:{hashlib.sha256(query.encode()).hexdigest()}"

    if use_cache:
        cached = QUERY_CACHE.get(query_key)
        if cached is not None:
            return cached

    try:
        init_ns = dict(g.namespaces())
        prepared = prepareQuery(query, initNs=init_ns)
    except Exception as e:
        if _is_client_query_error(e):
            raise _query_error(e) from e
        raise

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
    except Exception as e:
        if _is_client_query_error(e):
            raise _query_error(e) from e
        raise

    if use_cache:
        QUERY_CACHE.put(query_key, results, _result_rows(results))
    if duration > 0.5:
        logger.info(f"SPARQL query took {duration:.2f}s (hash: {query_key[:16]}...)")
    return results


# ---------------------------------------------------------------------------
# Execution of client-supplied SPARQL queries (/sparql)
#
# Every query runs in its own child process, forked from a dedicated executor
# thread. The child
#   - inherits the loaded graph copy-on-write (nothing is reloaded or pickled),
#   - is killed by the kernel if that thread or the worker process dies
#     (PR_SET_PDEATHSIG), so it cannot outlive its supervisor,
#   - closes every inherited file descriptor except its result pipe,
#   - gets an address-space limit (RLIMIT_AS) of its starting size plus
#     SPARQL_MAX_QUERY_MEMORY_MB, never above an inherited limit.
# If any of these cannot be set up, the query is refused (503).
#
# One deadline (SPARQL_TIMEOUT_SECONDS) covers evaluation, serialization, the
# transfer of the result and waiting for the child to exit; after that the
# child is killed and reaped within a short grace period (_KILL_GRACE_SECONDS).
#
# Admission runs on the event loop before any thread or process is used: at
# most SPARQL_MAX_CONCURRENT_QUERIES execute, at most SPARQL_MAX_WAITING_QUERIES
# wait (for at most SPARQL_MAX_WAIT_SECONDS), everything else gets 503 at once.
# A slot is released when the execution finishes, not when the request ends,
# so a cancelled request cannot free a slot while its query is still running.
# All limits are per worker process. Linux only; elsewhere /sparql answers 503
# unless SPARQL_ALLOW_UNISOLATED is set (development only).
# ---------------------------------------------------------------------------


class SparqlLimitError(Exception):
    """Base class for query rejections caused by a resource limit."""

    status_code = 503


class SparqlTimeoutError(SparqlLimitError):
    """The query did not finish within SPARQL_TIMEOUT_SECONDS."""


class SparqlMemoryError(SparqlLimitError):
    """The query process ran out of its memory allowance."""


class SparqlResultTooLargeError(SparqlLimitError):
    """The result exceeds SPARQL_MAX_RESULT_ROWS or SPARQL_MAX_RESPONSE_BYTES."""

    status_code = 400


class SparqlBusyError(SparqlLimitError):
    """No execution slot is free and the waiting queue is full."""


class SparqlUnavailableError(Exception):
    """Public query execution is disabled or a query process could not be set up."""


class SparqlInternalError(Exception):
    """The query process failed for a reason that is not the client's fault."""


_HEADER_MAX_BYTES = 4096
_KILL_GRACE_SECONDS = 2.0
_IS_LINUX = sys.platform.startswith("linux")
_LIBC = ctypes.CDLL(None, use_errno=True) if _IS_LINUX else None
_PR_SET_PDEATHSIG = 1


def _sparql_isolation_available() -> bool:
    return _IS_LINUX and hasattr(os, "fork")


if not _sparql_isolation_available():
    if settings.SPARQL_ALLOW_UNISOLATED:
        logger.warning(
            "SPARQL queries will run in-process WITHOUT time or memory limits "
            "(SPARQL_ALLOW_UNISOLATED=true). Do not use this setting in production."
        )
    else:
        logger.error(
            "Query isolation needs Linux; /sparql will answer 503. "
            "Set SPARQL_ALLOW_UNISOLATED=true for local development only."
        )


class _SparqlAdmission:
    """Admission to query execution. Slots belong to executions, not to requests.

    Used on the event loop. Queries run on a dedicated executor with exactly
    SPARQL_MAX_CONCURRENT_QUERIES threads; a slot is released from the
    executor's done-callback when the blocking function has returned (or was
    cancelled before it started), never merely because the awaiting request
    was cancelled.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sem: Optional[asyncio.Semaphore] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._waiting = 0

    def reset(self) -> None:
        """Start afresh (picks up changed settings; used by tests)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._loop, self._sem, self._executor, self._waiting = None, None, None, 0

    def _ensure(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Semaphore, ThreadPoolExecutor]:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self.reset()
            n = settings.SPARQL_MAX_CONCURRENT_QUERIES
            self._loop = loop
            self._sem = asyncio.Semaphore(n)
            self._executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="sparql-query")
        assert self._sem is not None and self._executor is not None
        return loop, self._sem, self._executor

    async def _acquire(self, sem: asyncio.Semaphore) -> None:
        if not sem.locked():
            await sem.acquire()
            return
        if self._waiting >= settings.SPARQL_MAX_WAITING_QUERIES:
            raise SparqlBusyError("The SPARQL endpoint is busy; try again later")
        self._waiting += 1
        try:
            await asyncio.wait_for(sem.acquire(), timeout=settings.SPARQL_MAX_WAIT_SECONDS)
        except TimeoutError:
            raise SparqlBusyError("The SPARQL endpoint is busy; try again later")
        finally:
            self._waiting -= 1

    @staticmethod
    def _release_from_thread(loop: asyncio.AbstractEventLoop, sem: asyncio.Semaphore) -> None:
        try:
            loop.call_soon_threadsafe(sem.release)
        except RuntimeError:
            pass  # event loop already closed: nothing left to admit

    async def run(self, fn: Any, *args: Any) -> Any:
        loop, sem, executor = self._ensure()
        await self._acquire(sem)
        # No await between acquiring and submitting: cancellation cannot leak the slot.
        try:
            cfut = executor.submit(fn, *args)
        except BaseException:
            sem.release()
            raise
        cfut.add_done_callback(lambda _f: self._release_from_thread(loop, sem))
        return await asyncio.wrap_future(cfut)


SPARQL_ADMISSION = _SparqlAdmission()


def _execute_and_serialize(query: str, accept: str) -> tuple[bytes, str]:
    """Run a client query (never cached), enforce the result limits, serialize it."""
    results = run_sparql_query(query, use_cache=False)
    rows = _result_rows(results)
    max_rows = settings.SPARQL_MAX_RESULT_ROWS
    if rows > max_rows:
        raise SparqlResultTooLargeError(
            f"Query result has {rows} rows/triples, more than the limit of {max_rows}. "
            "Add a LIMIT clause or narrow the query."
        )
    response = format_sparql_results(results, accept)
    body = bytes(response.body)
    max_bytes = settings.SPARQL_MAX_RESPONSE_BYTES
    if len(body) > max_bytes:
        raise SparqlResultTooLargeError(
            f"Serialized result is {len(body)} bytes, more than the limit of {max_bytes}. "
            "Add a LIMIT clause or narrow the query."
        )
    return body, str(response.media_type)


# --- set-up steps run in the child; any failure refuses the query ------------


def _set_parent_death_signal(parent_pid: int) -> None:
    """Ask the kernel to SIGKILL this child when its parent thread/process dies."""
    if _LIBC is None or _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno() if _LIBC is not None else 0, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:  # parent died before prctl took effect
        os._exit(1)


def _close_inherited_fds(keep: int) -> None:
    """Close every inherited descriptor above stderr except `keep` (Linux: /proc/self/fd)."""
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd > 2 and fd != keep:
            try:
                os.close(fd)
            except OSError:
                pass  # e.g. the descriptor of the listing itself, already closed


def _limit_child_memory(extra_mb: int) -> None:
    """Cap the address space at the current size + extra_mb, never above an inherited limit.

    The child starts as a copy of the server (graph included), so the limit is
    relative to the address space it inherited. Note that RLIMIT_AS limits
    address space, not physical memory: copy-on-write of inherited pages can
    add up to the inherited size in physical memory on top of the allowance.
    """
    with open("/proc/self/statm") as f:
        current = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    old_soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    soft = current + extra_mb * 1024 * 1024
    for inherited in (old_soft, hard):
        if inherited != resource.RLIM_INFINITY:
            soft = min(soft, inherited)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _encode_header(header: dict[str, Any]) -> bytes:
    """Serialize a result header so that it always fits in _HEADER_MAX_BYTES.

    Error messages can contain parts of the query (up to SPARQL_MAX_QUERY_LENGTH
    characters), so a long message is truncated. json.dumps escapes non-ASCII
    characters as \\uXXXX, and characters outside the Basic Multilingual Plane
    as a surrogate pair: at most 12 bytes per character, which bounds the
    encoded size of the truncated message.
    """
    encoded = json.dumps(header).encode()
    if len(encoded) < _HEADER_MAX_BYTES:
        return encoded
    header = dict(header)
    message = str(header.get("message", ""))
    keep = (_HEADER_MAX_BYTES - 512) // 12
    header["message"] = message[:keep] + " ... (message truncated)"
    encoded = json.dumps(header).encode()
    if len(encoded) < _HEADER_MAX_BYTES:
        return encoded
    return json.dumps({"kind": "internal", "message": "header too large"}).encode()


def _write_result(write_fd: int, header: dict[str, Any], body: bytes = b"") -> None:
    with os.fdopen(write_fd, "wb") as out:
        out.write(_encode_header(header) + b"\n")
        if body:
            out.write(body)


def _sparql_child_main(write_fd: int, query: str, accept: str, parent_pid: int) -> None:
    """Body of the forked child: apply the resource limits, run the query, report, exit."""
    exit_code = 0
    try:
        try:
            _set_parent_death_signal(parent_pid)  # first: protect against an orphaned child early
            gc.freeze()  # never finalize objects inherited from the server
            _close_inherited_fds(keep=write_fd)
            _limit_child_memory(settings.SPARQL_MAX_QUERY_MEMORY_MB)
        except Exception as e:
            _write_result(write_fd, {"kind": "setup_failed", "message": f"{type(e).__name__}: {e}"})
            return
        body = b""
        try:
            body, media_type = _execute_and_serialize(query, accept)
            header: dict[str, Any] = {"kind": "ok", "media_type": media_type, "length": len(body)}
        except SparqlResultTooLargeError as e:
            header = {"kind": "too_large", "message": str(e)}
        except SparqlQueryError as e:
            header = {"kind": "query_error", "message": e.message}
        except MemoryError:
            body = b""
            header = {"kind": "memory", "message": "Query exceeded the memory limit"}
        except GraphNotInitializedError:
            header = {"kind": "no_graph", "message": "RDF graph not initialized"}
        except Exception as e:
            header = {"kind": "internal", "message": type(e).__name__}
        _write_result(write_fd, header, body)
    except BaseException:
        exit_code = 1
    finally:
        # Leave immediately: never run the server's atexit handlers or finalizers.
        os._exit(exit_code)


# --- supervision in the parent -----------------------------------------------

_UNREAPED: set[int] = set()
_UNREAPED_LOCK = threading.Lock()


def _reap_leftovers() -> None:
    """Reap children that could not be reaped within the grace period earlier."""
    with _UNREAPED_LOCK:
        for pid in list(_UNREAPED):
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done = pid
            if done:
                _UNREAPED.discard(pid)


# Wait status for a child that is no longer waitable by us (reaped elsewhere):
# it is gone, but how it ended is unknown, so it never counts as a clean exit.
_STATUS_UNKNOWN = -1


def _wait_child(pid: int, until: float) -> Optional[int]:
    """Wait for the child without blocking past `until`.

    Returns its wait status, _STATUS_UNKNOWN if it can no longer be waited for,
    or None if it is still running at `until`. The deadline is checked between
    polls (every ~2 ms, subject to scheduling), so it is not exact to the
    millisecond.
    """
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return _STATUS_UNKNOWN
        if done:
            return status
        if time.monotonic() >= until:
            return None
        time.sleep(0.002)


def _finish_child(pid: int, deadline: float, kill: bool) -> tuple[Optional[int], bool]:
    """Make sure the child is gone.

    Returns (wait status, _STATUS_UNKNOWN, or None if it could not be reaped;
    whether forced cleanup was started). With kill=False the child gets until the
    deadline to exit by itself.
    """
    if not kill:
        # The pipe is closed; the child should exit at once, but EOF is not exit.
        status = _wait_child(pid, deadline)
        if status is not None:
            return status, False
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    status = _wait_child(pid, time.monotonic() + _KILL_GRACE_SECONDS)
    if status is None:
        with _UNREAPED_LOCK:
            _UNREAPED.add(pid)
        logger.error("SPARQL query process %d did not exit after SIGKILL; will reap later", pid)
    return status, True


class _Deadline:
    def __init__(self, fd: int, deadline: float, timeout: float):
        self.deadline = deadline
        self.timeout = timeout
        self._poller = select.poll()  # poll(), not select(): server fds can exceed FD_SETSIZE
        self._poller.register(fd, select.POLLIN | select.POLLHUP)

    def wait_readable(self) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not self._poller.poll(remaining * 1000):
            raise SparqlTimeoutError(f"SPARQL query exceeded the time limit of {self.timeout:g} s")


def _read_child_result(fd: int, clock: _Deadline, max_body: int) -> tuple[Optional[dict], bytes]:
    """Read `header-json \\n body` from the child. Returns (None, b"") if the child
    died before sending a complete header. Enforces the body limit exactly and
    keeps at most one copy of the body while reading."""
    buf = bytearray()
    while b"\n" not in buf:
        clock.wait_readable()
        chunk = os.read(fd, 65536)
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > _HEADER_MAX_BYTES and b"\n" not in buf[:_HEADER_MAX_BYTES]:
            raise SparqlInternalError("SPARQL query process sent a malformed result")
    header_line, _, rest = bytes(buf).partition(b"\n")
    del buf
    try:
        header = json.loads(header_line)
        kind = header["kind"]
    except (ValueError, KeyError, TypeError):
        raise SparqlInternalError("SPARQL query process sent a malformed result")

    length = header.get("length", 0) if kind == "ok" else 0
    if not isinstance(length, int) or length < 0:
        raise SparqlInternalError("SPARQL query process sent a malformed result")
    if length > max_body:
        raise SparqlResultTooLargeError(
            f"Serialized result is {length} bytes, more than the limit of {max_body}"
        )
    if len(rest) > length:
        raise SparqlInternalError("SPARQL query process sent more data than announced")

    body = bytearray(length)
    view = memoryview(body)
    view[: len(rest)] = rest
    pos = len(rest)
    del rest
    while pos < length:
        clock.wait_readable()
        n = os.readv(fd, [view[pos:]])
        if n == 0:
            return None, b""  # died half-way through the body
        pos += n
    clock.wait_readable()
    if os.read(fd, 1):
        raise SparqlInternalError("SPARQL query process sent more data than announced")
    view.release()
    return header, bytes(body)


def _run_sparql_in_child(query: str, accept: str) -> tuple[bytes, str]:
    """Run one client query in a forked child under a single end-to-end deadline.

    A result (of any kind) is used only if the child has also exited normally,
    with status 0, by the deadline; otherwise the query fails (503 if it had to
    be killed, 500 for any other abnormal exit).
    """
    _reap_leftovers()
    with _UNREAPED_LOCK:
        stuck = len(_UNREAPED)
    if stuck:
        # A process that survived SIGKILL no longer holds a slot; refusing new
        # queries until it is gone keeps the number of query processes within
        # SPARQL_MAX_CONCURRENT_QUERIES.
        raise SparqlUnavailableError(
            "SPARQL queries are temporarily refused: an earlier query process has not exited yet"
        )
    timeout = settings.SPARQL_TIMEOUT_SECONDS
    deadline = time.monotonic() + timeout
    parent_pid = os.getpid()

    try:
        read_fd, write_fd = os.pipe()
    except OSError as e:
        raise SparqlUnavailableError(f"Could not start a query process ({e.strerror})")
    try:
        pid = os.fork()
    except OSError as e:
        os.close(read_fd)
        os.close(write_fd)
        raise SparqlUnavailableError(f"Could not start a query process ({e.strerror})")
    if pid == 0:  # child: whatever happens, never return into the server's code
        try:
            os.close(read_fd)
            _sparql_child_main(write_fd, query, accept, parent_pid)
        finally:
            os._exit(1)
    os.close(write_fd)

    complete = False
    try:
        header, body = _read_child_result(read_fd, _Deadline(read_fd, deadline, timeout),
                                          settings.SPARQL_MAX_RESPONSE_BYTES)
        complete = True
    finally:
        os.close(read_fd)
        status, killed = _finish_child(pid, deadline, kill=not complete)

    known = status is not None and status != _STATUS_UNKNOWN
    exited_cleanly = known and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    if header is not None:
        # The child reported; the report counts only if it then exited normally
        # by the deadline. Forced cleanup means the deadline was missed, even if
        # the child happened to exit normally just before the SIGKILL.
        if killed:
            raise SparqlTimeoutError(f"SPARQL query exceeded the time limit of {timeout:g} s")
        if not exited_cleanly:
            logger.error("SPARQL query process exited abnormally after reporting (status %s)", status)
            raise SparqlInternalError("SPARQL query failed because of a server error")

    kind = header.get("kind") if header else None
    if kind == "ok":
        return body, header["media_type"]
    if kind == "query_error":
        raise SparqlQueryError(header["message"])
    if kind == "too_large":
        raise SparqlResultTooLargeError(header["message"])
    if kind == "memory":
        raise SparqlMemoryError(header["message"])
    if kind == "no_graph":
        raise GraphNotInitializedError()
    if kind == "setup_failed":
        logger.error("SPARQL query process could not be resource-limited: %s", header.get("message"))
        raise SparqlUnavailableError("SPARQL query refused: the query process could not be resource-limited")
    if kind == "internal":
        logger.error("SPARQL query process failed: %s", header.get("message"))
        raise SparqlInternalError("SPARQL query failed because of a server error")
    # No complete result: the child died before it could report.
    if status == _STATUS_UNKNOWN:
        logger.error("SPARQL query process %d ended without a result; exit status unknown", pid)
        raise SparqlInternalError("SPARQL query failed because of a server error")
    if status is None or os.WIFSIGNALED(status):
        logger.warning("SPARQL query process terminated (status %s)", status)
        raise SparqlMemoryError("SPARQL query process was terminated; it probably exceeded the memory limit")
    logger.error("SPARQL query process exited abnormally (status %d)", status)
    raise SparqlInternalError("SPARQL query failed because of a server error")


def _run_public_sparql_blocking(query: str, accept: str) -> tuple[bytes, str]:
    if _sparql_isolation_available():
        return _run_sparql_in_child(query, accept)
    if settings.SPARQL_ALLOW_UNISOLATED:
        try:
            return _execute_and_serialize(query, accept)
        except (SparqlQueryError, SparqlResultTooLargeError, GraphNotInitializedError):
            raise
        except MemoryError:
            raise SparqlMemoryError("Query exceeded the available memory")
        except Exception:
            logger.exception("SPARQL query failed in-process")
            raise SparqlInternalError("SPARQL query failed because of a server error")
    raise SparqlUnavailableError("SPARQL queries are disabled: query isolation needs Linux")


async def execute_public_sparql(query: str, accept: str) -> Response:
    """Run a client-supplied SPARQL query without blocking the event loop.

    Admission is decided on the event loop before any thread or process is
    used, so the number of running and waiting queries is bounded. Client
    queries never read or fill QUERY_CACHE. If the client disconnects, the
    query still runs until it finishes or reaches the time limit, and keeps
    its slot until then.
    """
    try:
        body, media_type = await SPARQL_ADMISSION.run(_run_public_sparql_blocking, query, accept)
    except GraphNotInitializedError:
        raise HTTPException(status_code=503, detail="RDF graph not initialized")
    except SparqlQueryError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except SparqlBusyError as e:
        raise HTTPException(status_code=503, detail=str(e), headers={"Retry-After": "5"})
    except SparqlLimitError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except SparqlUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SparqlInternalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(body, media_type=media_type)


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
        const_obj = _extract_constant(g, c_uri, include_details=True)
        if const_obj is not None:
            all_vals = ([const_obj.value] if const_obj.value else []) + (const_obj.historicalValues or [])
            all_vals.sort(key=lambda v: v.versionId, reverse=True)
            const_obj.value = cv_obj
            const_obj.historicalValues = all_vals
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

    return await execute_public_sparql(query, accept)


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

    accept = _resolve_accept(request)
    return await execute_public_sparql(query, accept)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
