from decimal import Decimal
import json
import math
import re
import socket
import threading
from fastapi.testclient import TestClient
import pytest
import rdflib
from src.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_json_and_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to CODATA DRUM Physical Fundamental Constants API!"
    assert response.json()["version"] == "0.5.0"

    html_resp = client.get("/", headers={"Accept": "text/html"})
    assert html_resp.status_code == 200
    assert "Developer API Sandbox" in html_resp.text
    assert "/explorer" in html_resp.text
    assert "postman.com" not in html_resp.text
    assert "/playground" not in html_resp.text


def test_active_navigation_highlighting(client):
    """Verify that each page highlights only its corresponding navigation link."""
    import re

    routes = [
        ("/", "Developer Portal"),
        ("/explorer", "Explorer (Viewer)"),
        ("/constants", "Constants"),
        ("/constants/FaradayConstant", "Constants"),
        ("/constants/versions", "Versions"),
        ("/quantities", "Quantities"),
        ("/concepts", "Concepts"),
        ("/units", "Units"),
        ("/sparql", "SPARQL"),
    ]
    for path, expected_label in routes:
        resp = client.get(path, headers={"Accept": "text/html"})
        assert resp.status_code == 200
        nav_match = re.search(r"<nav.*?</nav>", resp.text, re.DOTALL)
        assert nav_match, f"No nav element found in response for {path}"
        actives = re.findall(r"<a [^>]*class=\"active\"[^>]*>(.*?)</a>", nav_match.group(0))
        assert actives == [expected_label], f"Expected [{expected_label}] active for {path}, got {actives}"


def test_compact_number_filter_and_large_number_preservation(client):
    """Verify that scientific numbers preserve their exponent in compact view and HTML."""
    from src.app import _format_compact_number

    # Large and small numbers with exponents must never lose the exponent
    assert _format_compact_number("1.356392489652132101257636096714128e50") == "1.35639248965213…e50"
    assert _format_compact_number("1.05457181764615639126242800330228e-34") == "1.0545718176461…e-34"
    assert _format_compact_number("6.6446573450e-27") == "6.6446573450e-27"
    assert _format_compact_number("96485.3321233100184") == "96485.3321233100184"

    # HTML table should render compact numbers with exponents preserved
    resp = client.get("/constants", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "e-27" in resp.text or "e-34" in resp.text


def test_explorer_history_navigation_support(client):
    """Verify that Knowledge Explorer contains browser history navigation and popstate handling."""
    resp = client.get("/explorer", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "updateBrowserUrl" in resp.text
    assert "window.history.pushState" in resp.text
    assert "window.history.replaceState" in resp.text
    assert "popstate" in resp.text


def test_explorer_search_scope_options_and_filtering(client):
    """Verify that Knowledge Explorer provides search scope selector and scopes match accurately."""
    resp = client.get("/explorer", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    # Search scope dropdown & options
    assert 'id="search-scope"' in resp.text
    assert 'value="all"' in resp.text
    assert 'value="name"' in resp.text
    assert 'value="symbol"' in resp.text
    assert 'value="formula"' in resp.text
    assert 'value="id"' in resp.text
    # Clear button & search helpers
    assert 'id="search-clear-btn"' in resp.text
    assert "handleSearchInput" in resp.text
    assert "clearSearch" in resp.text
    assert "updateSearchPlaceholder" in resp.text
    assert "getFieldValuesForScope" in resp.text

    # Verify scope filtering logic against actual API constant dataset
    const_resp = client.get("/constants", headers={"Accept": "application/json"})
    assert const_resp.status_code == 200
    constants = const_resp.json()

    def filter_by_scope(items, q, scope):
        tokens = q.lower().split()
        res = []
        for item in items:
            if scope == "name":
                fields = [item.get("name"), item.get("nameFr")]
            elif scope == "symbol":
                fields = [item.get("symbolTex"), item.get("symbolHtml")]
            elif scope == "formula":
                fields = [item.get("formula"), item.get("formulaLatex"), item.get("siExpression")]
            elif scope == "id":
                fields = [item.get("id"), item.get("uri"), item.get("ucumCode")]
            else:
                fields = [
                    item.get("id"),
                    item.get("name"),
                    item.get("nameFr"),
                    item.get("definition"),
                    item.get("symbolTex"),
                    item.get("symbolHtml"),
                    item.get("formula"),
                    item.get("formulaLatex"),
                    item.get("ucumCode"),
                    item.get("siExpression"),
                ]
            hay = " ".join(filter(None, fields)).lower()
            if all(t in hay for t in tokens):
                res.append(item)
        return res

    all_boltz = filter_by_scope(constants, "boltz", "all")
    name_boltz = filter_by_scope(constants, "boltz", "name")
    formula_boltz = filter_by_scope(constants, "boltz", "formula")
    symbol_boltz = filter_by_scope(constants, "boltz", "symbol")

    # Name-only scope must isolate only the 5 Boltzmann constants
    assert len(name_boltz) == 5
    assert all("boltzmann" in (c["name"] or "").lower() for c in name_boltz)

    # Broad scope includes derivations with Boltzmann in formula
    assert len(all_boltz) == 26
    assert len(formula_boltz) == 25
    assert len(symbol_boltz) == 0


def test_historical_evolution_navigation_support(client):
    """Verify that Historical Evolution table allows interactive navigation to previous releases."""
    resp = client.get("/explorer", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "selectConstantVersion" in resp.text
    assert "historical-view-banner" in resp.text
    assert "active-history-row" in resp.text
    assert "btn-version-pill" in resp.text
    assert "btn-inspect-version" in resp.text
    assert "Actions" in resp.text
    assert "Release &amp; Role" in resp.text
    assert "● Active" not in resp.text

    # Verify standalone historical constant value page has navigation support
    hist_resp = client.get("/constants/PlanckConstant/2018", headers={"Accept": "text/html"})
    assert hist_resp.status_code == 200
    assert "Historical CODATA 2018 Release Record" in hist_resp.text
    assert "View Current Recommended (2022)" in hist_resp.text
    assert "explorer?entity=constants&id=PlanckConstant&version=2018" in hist_resp.text
    assert "Active View" in hist_resp.text


def test_footer_and_header_community_links_and_branding(client):
    """Verify feedback button, contact email, DRUM and CODATA branding/logos are rendered across pages."""
    # Test static assets availability
    logo_drum_header = client.get("/static/drum_logo_750x500_white.png")
    assert logo_drum_header.status_code == 200
    assert "image/png" in logo_drum_header.headers["content-type"]

    logo_drum_footer = client.get("/static/drum_logo_750x500.png")
    assert logo_drum_footer.status_code == 200
    assert "image/png" in logo_drum_footer.headers["content-type"]

    logo_codata = client.get("/static/codata_logo.svg")
    assert logo_codata.status_code == 200
    assert "image/svg+xml" in logo_codata.headers["content-type"]

    for path in ["/", "/explorer", "/constants", "/sparql"]:
        resp = client.get(path, headers={"Accept": "text/html"})
        assert resp.status_code == 200
        # GitHub issue feedback button
        assert "https://github.com/codata/drum-constants-api/issues/new" in resp.text
        # DRUM contact email
        assert "mailto:drum@codata.org" in resp.text
        # DRUM website link
        assert "https://drum.codata.org" in resp.text
        # CODATA website link in footer
        assert "https://codata.org" in resp.text
        # Header DRUM white logo
        assert "drum_logo_750x500_white.png" in resp.text
        # Footer DRUM standard logo
        assert "drum_logo_750x500.png" in resp.text
        # Footer CODATA logo
        assert "codata_logo.svg" in resp.text


def test_header_no_wrap_styling(client):
    """Verify that header styles enforce single-line no-wrap layout to optimize vertical space."""
    resp = client.get("/", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    # Header row must have flex-wrap: nowrap
    assert ".header-row {" in resp.text
    assert "flex-wrap: nowrap;" in resp.text
    # Header actions must not wrap or shrink
    assert "flex-shrink: 0;" in resp.text
    assert "white-space: nowrap;" in resp.text
    # Optimized compact padding (not 1.5rem)
    assert "padding: 0.75rem 0;" in resp.text


def test_static_files_included_in_pyproject_build():
    """Verify pyproject.toml includes src/static/**/* in hatch build targets."""
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    includes = data.get("tool", {}).get("hatch", {}).get("build", {}).get("include", [])
    assert "src/static/**/*" in includes, "src/static/**/* must be included in hatch build targets"


def test_api_version(client):
    """Verify package and OpenAPI version is 0.5.0."""
    from src import __version__

    assert __version__ == "0.5.0"
    openapi = client.get("/openapi.json").json()
    assert openapi["info"]["version"] == "0.5.0"


def test_health_and_ontology_endpoints(client):
    """Verify GET /health and GET /ontology."""
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert h_data["apiVersion"] == "0.5.0"
    assert h_data["constantsTotal"] == 359
    assert h_data["constantsActive"] == 355
    assert h_data["computedUpgradesTotal"] == 124

    ont_ttl = client.get("/ontology", headers={"Accept": "text/turtle"})
    assert ont_ttl.status_code == 200
    assert "text/turtle" in ont_ttl.headers["content-type"]
    assert "codata:adjustmentRole" in ont_ttl.text

    ont_jsonld = client.get("/ontology?format=jsonld")
    assert ont_jsonld.status_code == 200
    assert "application/ld+json" in ont_jsonld.headers["content-type"]


def test_explorer_and_playground_removed(client):
    """Verify GET /explorer renders the read-only viewer and /playground is removed (404)."""
    exp_resp = client.get("/explorer")
    assert exp_resp.status_code == 200
    assert "Knowledge Explorer" in exp_resp.text
    assert "Two-Source Computed Precision Upgrade" in exp_resp.text

    assert client.get("/playground").status_code == 404
    assert client.get("/playground/sparql").status_code == 404


def test_openapi_contract_has_response_models(client):
    """Verify OpenAPI schema exposes typed response models rather than untyped Any."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    schemas = spec.get("components", {}).get("schemas", {})
    for model_name in ("Concept", "Quantity", "Constant", "ConstantValue", "Unit", "Version"):
        assert model_name in schemas, f"Missing {model_name} in OpenAPI components.schemas"


def test_constants_browse_excludes_deprecated_by_default(client):
    """Verify GET /constants returns only active 2022 constants by default."""
    response = client.get("/constants")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 355

    deprecated_ids = {
        "FaradayConstantConventionalElectricCurrent",
        "MolarPlanckConstantTimesC",
        "TritonElectronMagneticMomentRatio",
        "TritonNeutronMagneticMomentRatio",
    }
    returned_ids = {c["id"] for c in data}
    assert returned_ids.isdisjoint(deprecated_ids)
    for c in data:
        assert c["value"]["versionId"] == "2022"
        assert c["value"]["adjustmentRole"] in {
            "defining",
            "computed_exact",
            "adjusted",
            "computed_derived",
        }


def test_constants_browse_filters(client):
    """Verify rich filtering on GET /constants."""
    # 1. Include deprecated
    response = client.get("/constants?include_deprecated=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 359
    by_id = {c["id"]: c for c in data}
    assert by_id["FaradayConstantConventionalElectricCurrent"].get("isDeprecated") is True

    # 2. Version filter (2014)
    resp_2014 = client.get("/constants?version=2014")
    assert resp_2014.status_code == 200
    data_2014 = resp_2014.json()
    assert len(data_2014) == 335
    assert all(c["value"]["versionId"] == "2014" for c in data_2014)

    # 3. Role filter in 2022 vs 2014
    defining_2022 = client.get("/constants?role=defining").json()
    assert len(defining_2022) == 12
    defining_2014 = client.get("/constants?version=2014&role=defining").json()
    assert len(defining_2014) == 9

    comp_exact_2022 = client.get("/constants?role=computed_exact").json()
    assert len(comp_exact_2022) == 69

    # 4. Value source & storage type filters
    upgraded_2022 = client.get("/constants?value_source=computed").json()
    assert len(upgraded_2022) == 62
    decimal_2022 = client.get("/constants?recommended_numeric_type=decimal").json()
    assert len(decimal_2022) == 61

    # 5. Search query filter
    search_faraday = client.get("/constants?q=faraday").json()
    assert any(c["id"] == "FaradayConstant" for c in search_faraday)


def test_computable_constants_two_source_and_dag_in_api(client):
    """Verify two-source precision, formulas, DAG links, and NIST alias lookup."""
    # Lookup FaradayConstant
    f_resp = client.get("/constants/FaradayConstant")
    assert f_resp.status_code == 200
    f_data = f_resp.json()
    assert f_data["symbolTex"] == "F"
    assert f_data["formula"] == "AvogadroConstant * ElementaryCharge"
    assert f_data["formulaLatex"] == r"N_{\text{A}} e"
    dep_ids = [d["id"] for d in f_data["derivedFrom"]]
    assert dep_ids == ["AvogadroConstant", "ElementaryCharge"]
    ident_map = {i["id"]: i["value"] for i in f_data["identifiers"]}
    assert ident_map["NIST"] == "f"

    f_val = f_data["value"]
    assert f_val["versionId"] == "2022"
    assert f_val["adjustmentRole"] == "computed_exact"
    assert f_val["isComputed"] is True
    assert f_val["valueSource"] == "computed"
    assert f_val["value"] == "96485.3321233100184"
    assert f_val["nistValue"] == "96485.33212"
    assert f_val["isNistTruncated"] is True
    assert f_val.get("isTruncated") is None
    assert f_val["significantDigits"] == 18
    assert f_val["fitsFloat64"] is False
    assert f_val["recommendedNumericType"] == "decimal"

    # Lookup ReducedPlanckConstant via NIST alias 'hbar'
    hbar_resp = client.get("/constants/hbar")
    assert hbar_resp.status_code == 200
    hbar_data = hbar_resp.json()
    assert hbar_data["id"] == "ReducedPlanckConstant"
    assert hbar_data["value"]["valueSource"] == "computed"
    assert hbar_data["value"]["nistValue"] == "1.054571817e-34"
    assert hbar_data["value"]["isTruncated"] is True
    assert hbar_data["value"]["isNistTruncated"] is True
    assert len(hbar_data["value"]["valueDecimal"]) >= 34

    # Verify PlanckConstant has downstream dependents in usedIn
    h_data = client.get("/constants/PlanckConstant").json()
    used_in_ids = {u["id"] for u in h_data.get("usedIn", [])}
    assert "ReducedPlanckConstant" in used_in_ids
    assert "JosephsonConstant" in used_in_ids


def test_metrological_defining_and_derived_constants(client):
    """Verify exact SI defining constants and derived constants in 2022 via API."""
    c_resp = client.get("/constants/SpeedOfLight", headers={"Accept": "application/json"})
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert c_data["value"]["versionId"] == "2022"
    assert c_data["value"]["value"] == "299792458"
    assert c_data["value"]["valueFloat"] == 299792458.0
    assert c_data["value"]["isExact"] is True
    assert c_data["value"]["adjustmentRole"] == "defining"
    assert c_data["value"]["isComputed"] is False
    assert "uncertainty" not in c_data["value"]

    h_resp = client.get("/constants/PlanckConstant", headers={"Accept": "application/json"})
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["value"]["isExact"] is True
    assert h_data["value"]["adjustmentRole"] == "defining"
    assert math.isclose(h_data["value"]["valueFloat"], 6.62607015e-34, rel_tol=1e-15)
    assert not h_data["value"]["valueDecimal"].startswith("0.0000000000")
    assert h_data["value"]["nistExponent"] == -34

    e_data = client.get("/constants/ElementaryCharge").json()
    k_data = client.get("/constants/BoltzmannConstant").json()
    na_data = client.get("/constants/AvogadroConstant").json()
    r_data = client.get("/constants/MolarGasConstant").json()

    assert e_data["value"]["isExact"] is True
    assert k_data["value"]["isExact"] is True
    assert na_data["value"]["isExact"] is True
    assert r_data["value"]["isExact"] is True
    assert r_data["value"]["adjustmentRole"] == "computed_exact"
    assert math.isclose(
        r_data["value"]["valueFloat"],
        k_data["value"]["valueFloat"] * na_data["value"]["valueFloat"],
        rel_tol=1e-9,
    )

    alpha_data = client.get("/constants/FineStructureConstant").json()
    rydberg_data = client.get("/constants/RydbergConstant").json()
    assert alpha_data["value"]["isExact"] is False
    assert alpha_data["value"]["adjustmentRole"] == "adjusted"
    assert float(alpha_data["value"]["uncertaintyFloat"]) > 0
    assert rydberg_data["value"]["isExact"] is False
    assert float(rydberg_data["value"]["uncertaintyFloat"]) > 0


def test_constant_version_endpoint_and_si_role_transition(client):
    """Verify GET /constants/{id}/{versionId} reflects historical SI role shift."""
    h_1998 = client.get("/constants/PlanckConstant/1998").json()
    assert h_1998["versionId"] == "1998"
    assert h_1998["isExact"] is False
    assert h_1998["adjustmentRole"] == "adjusted"
    assert h_1998["isComputed"] is False
    assert "uncertainty" in h_1998

    h_2022 = client.get("/constants/PlanckConstant/2022").json()
    assert h_2022["versionId"] == "2022"
    assert h_2022["isExact"] is True
    assert h_2022["adjustmentRole"] == "defining"
    assert h_2022["isComputed"] is False

    mu0_2014 = client.get("/constants/ElectromagneticPermeabilityOfVacuum/2014").json()
    assert mu0_2014["adjustmentRole"] == "defining"
    assert mu0_2014["isComputed"] is False
    assert mu0_2014["isExact"] is True

    mu0_2022 = client.get("/constants/ElectromagneticPermeabilityOfVacuum/2022").json()
    assert mu0_2022["adjustmentRole"] == "computed_derived"
    assert mu0_2022["isComputed"] is True
    assert mu0_2022["isExact"] is False


def test_content_negotiation_jsonld_html_turtle(client):
    """Verify JSON-LD @context, HTML embedded JSON-LD, and high-precision Turtle serialization."""
    ld_resp = client.get("/constants/PlanckConstant", headers={"Accept": "application/ld+json"})
    assert ld_resp.status_code == 200
    assert "application/ld+json" in ld_resp.headers["content-type"]
    ld_json = ld_resp.json()
    assert "@context" in ld_json

    html_resp = client.get("/constants/PlanckConstant", headers={"Accept": "text/html"})
    assert html_resp.status_code == 200
    assert "Planck Constant" in html_resp.text
    match = re.search(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html_resp.text,
        re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert "@context" in embedded
    assert "adjustmentRole" in embedded["@context"]

    # Turtle format with xsd:double and 34-digit xsd:decimal round-trip verification
    ttl_resp = client.get("/constants/ReducedPlanckConstant", headers={"Accept": "text/turtle"})
    assert ttl_resp.status_code == 200
    assert ttl_resp.headers["content-type"] == "text/turtle; charset=utf-8"
    parsed_g = rdflib.Graph()
    parsed_g.parse(data=ttl_resp.text, format="turtle")
    model_ns = rdflib.Namespace("https://w3id.org/codata/model/")
    val_2022 = rdflib.URIRef("https://w3id.org/codata/constants/ReducedPlanckConstant/2022")
    val_flt = parsed_g.value(val_2022, model_ns.valueFloat)
    assert isinstance(val_flt, rdflib.Literal) and val_flt.datatype == rdflib.XSD.double
    val_dec = parsed_g.value(val_2022, model_ns.valueDecimal)
    val_str = parsed_g.value(val_2022, model_ns.value)
    assert Decimal(str(val_dec)) == Decimal(str(val_str))


def test_constant_detail_not_found(client):
    response = client.get("/constants/non_existent_constant_id_12345")
    assert response.status_code == 404


def test_concepts_and_quantities(client):
    c_resp = client.get("/concepts")
    assert c_resp.status_code == 200
    concepts = c_resp.json()
    assert len(concepts) > 50

    ev_resp = client.get("/concepts/ElectronVolt")
    assert ev_resp.status_code == 200
    assert ev_resp.json()["id"] == "ElectronVolt"

    q_resp = client.get("/quantities")
    assert q_resp.status_code == 200
    quantities = q_resp.json()
    assert len(quantities) > 20
    q_id = quantities[0]["id"]
    assert client.get(f"/quantities/{q_id}").status_code == 200

    ratio_qty = client.get("/quantities/ElectronProtonMassRatio").json()
    assert ratio_qty["isRatio"] is True


def test_units_enrichment_in_api(client):
    u_resp = client.get("/units")
    assert u_resp.status_code == 200
    units = u_resp.json()
    assert len(units) >= 60
    by_id = {u["id"]: u for u in units}
    assert by_id["J"]["name"] == "joule"
    assert by_id["J"]["ucumCode"] == "J"

    j_detail = client.get("/units/J").json()
    assert j_detail["id"] == "J"
    assert j_detail["name"] == "joule"
    assert j_detail["ucumCode"] == "J"


def test_versions_browse_and_detail(client):
    response = client.get("/constants/versions")
    assert response.status_code == 200
    data = response.json()
    version_ids = {v["id"] for v in data}
    assert {"1998", "2002", "2006", "2010", "2014", "2018", "2022"}.issubset(version_ids)

    v2022 = client.get("/constants/versions/2022")
    assert v2022.status_code == 200
    assert v2022.json()["id"] == "2022"
    assert len(v2022.json().get("constants", [])) == 355


def test_sparql_endpoint_and_browser_ui(client):
    # Browser HTML console when no query param is passed
    ui_resp = client.get("/sparql", headers={"Accept": "text/html"})
    assert ui_resp.status_code == 200
    assert "SPARQL 1.1 Query Console" in ui_resp.text

    # Protocol execution when query is provided
    query = """
    PREFIX codata: <https://w3id.org/codata/model/>
    PREFIX constant: <https://w3id.org/codata/constants/>
    SELECT ?val WHERE {
        constant:SpeedOfLight codata:hasValue ?v .
        ?v codata:versionId "2022" ;
           codata:value ?val .
    }
    """
    resp = client.get("/sparql", params={"query": query})
    assert resp.status_code == 200
    bindings = resp.json()["results"]["bindings"]
    assert len(bindings) == 1
    assert bindings[0]["val"]["value"] == "299792458"


# --- SPARQL SERVICE must not make the server contact other hosts (SSRF) ---


@pytest.fixture
def listener():
    """A local TCP listener that records whether anything connects to it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.2)
    hits = []
    stop = threading.Event()

    def accept_loop():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                hits.append(conn.recv(4096))
                conn.close()
            except OSError:
                pass

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}/probe", hits
    stop.set()
    t.join(1)
    srv.close()


SERVICE_QUERIES = [
    "SELECT * WHERE {{ SERVICE <{url}> {{ ?s ?p ?o }} }}",
    "select * where {{ service silent <{url}> {{ ?s ?p ?o }} }}",
    "SELECT * WHERE {{ ?a ?b ?c OPTIONAL {{ SERVICE <{url}> {{ ?s ?p ?o }} }} }} LIMIT 1",
    "CONSTRUCT {{ ?s ?p ?o }} WHERE {{ SERVICE <{url}> {{ ?s ?p ?o }} }}",
    "ASK {{ {{ ?a ?b ?c }} UNION {{ SERVICE <{url}> {{ ?s ?p ?o }} }} }}",
    "SELECT * WHERE {{ ?s ?p ?o MINUS {{ SERVICE <{url}> {{ ?s ?p ?o }} }} }}",
    "SELECT * WHERE {{ {{ SELECT * WHERE {{ SERVICE <{url}> {{ ?s ?p ?o }} }} }} }}",
    "SELECT * WHERE {{ ?s ?p ?o FILTER EXISTS {{ SERVICE <{url}> {{ ?s ?p ?o }} }} }}",
]


@pytest.mark.parametrize("template", SERVICE_QUERIES)
def test_sparql_get_refuses_service(client, listener, template):
    url, hits = listener
    r = client.get("/sparql", params={"query": template.format(url=url)})
    assert r.status_code == 400
    assert "SERVICE" in r.text
    assert hits == [], "server made an outbound request"


@pytest.mark.parametrize(
    "content_type",
    ["application/sparql-query", "application/x-www-form-urlencoded", "application/json"],
)
def test_sparql_post_refuses_service(client, listener, content_type):
    url, hits = listener
    q = SERVICE_QUERIES[0].format(url=url)
    if content_type == "application/sparql-query":
        r = client.post("/sparql", content=q, headers={"Content-Type": content_type})
    elif content_type == "application/json":
        r = client.post("/sparql", json={"query": q})
    else:
        r = client.post("/sparql", data={"query": q})
    assert r.status_code == 400
    assert hits == [], "server made an outbound request"


def test_sparql_syntax_error_is_400(client):
    r = client.get("/sparql", params={"query": "SELECT * WHERE { ?s ?p"})
    assert r.status_code == 400


def test_sparql_ordinary_query_still_works(client):
    q = (
        "PREFIX codata: <https://w3id.org/codata/model/> "
        "SELECT ?v WHERE { <https://w3id.org/codata/constants/SpeedOfLight/2022> codata:value ?v }"
    )
    r = client.get("/sparql", params={"query": q}, headers={"Accept": "application/sparql-results+json"})
    assert r.status_code == 200
    assert r.json()["results"]["bindings"][0]["v"]["value"] == "299792458"


def test_sparql_query_without_prefix_uses_graph_namespaces(client):
    """Test that queries omitting explicit PREFIX declarations resolve namespaces bound to the graph."""
    q = "SELECT ?v WHERE { constant:SpeedOfLight codata:hasValue ?val . ?val codata:versionId '2022' ; codata:value ?v }"
    r = client.get("/sparql", params={"query": q}, headers={"Accept": "application/sparql-results+json"})
    assert r.status_code == 200
    assert r.json()["results"]["bindings"][0]["v"]["value"] == "299792458"


def test_sparql_construct_query(client):
    """Test that CONSTRUCT queries format as RDF rather than throwing 500 serializer errors."""
    q = (
        "PREFIX codata: <https://w3id.org/codata/model/> "
        "PREFIX constant: <https://w3id.org/codata/constants/> "
        "CONSTRUCT { constant:SpeedOfLight codata:hasValue ?v } "
        "WHERE { constant:SpeedOfLight codata:hasValue ?v }"
    )
    r = client.get("/sparql", params={"query": q}, headers={"Accept": "text/turtle"})
    assert r.status_code == 200
    assert "text/turtle" in r.headers["content-type"]
    assert "SpeedOfLight" in r.text

    r_jsonld = client.get("/sparql", params={"query": q}, headers={"Accept": "application/ld+json"})
    assert r_jsonld.status_code == 200
    assert "application/ld+json" in r_jsonld.headers["content-type"]


def test_sparql_ask_query(client):
    """Test ASK queries."""
    q = "PREFIX constant: <https://w3id.org/codata/constants/> ASK { constant:SpeedOfLight ?p ?o }"
    r = client.get("/sparql", params={"query": q})
    assert r.status_code == 200
    assert r.json().get("boolean") is True


def test_sparql_from_clause_does_not_fetch_remote(client, listener):
    """Test that FROM <url> does not make an outbound network request."""
    url, hits = listener
    q = f"SELECT * FROM <{url}> WHERE {{ ?s ?p ?o }} LIMIT 1"
    r = client.get("/sparql", params={"query": q})
    assert r.status_code == 200
    assert hits == [], "server made an outbound request for FROM clause"
