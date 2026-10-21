import json
import math
import re
from fastapi.testclient import TestClient
from src.app import app
import pytest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to CODATA DRUM Physical Fundamental Constants API!"


def test_api_version(client):
    """Verify package and OpenAPI version is 0.4.0."""
    from src import __version__
    assert __version__ == "0.4.0"
    openapi = client.get("/openapi.json").json()
    assert openapi["info"]["version"] == "0.4.0"


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
    assert len(data) >= 350

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


def test_constants_browse_include_deprecated_and_version_filter(client):
    """Verify ?include_deprecated=true and ?version=2014 on GET /constants."""
    response = client.get("/constants?include_deprecated=true")
    assert response.status_code == 200
    data = response.json()
    by_id = {c["id"]: c for c in data}
    assert "FaradayConstantConventionalElectricCurrent" in by_id
    assert by_id["FaradayConstantConventionalElectricCurrent"].get("isDeprecated") is True

    resp_2014 = client.get("/constants?version=2014")
    assert resp_2014.status_code == 200
    data_2014 = resp_2014.json()
    assert len(data_2014) > 300
    assert all(c["value"]["versionId"] == "2014" for c in data_2014)


def test_metrological_defining_and_derived_constants(client):
    """Verify exact SI defining constants and derived constants in 2022 via API."""
    # 1. Speed of light in vacuum (exact)
    c_resp = client.get("/constants/SpeedOfLight", headers={"Accept": "application/json"})
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert c_data["value"]["versionId"] == "2022"
    assert c_data["value"]["value"] == "299792458"
    assert c_data["value"]["valueFloat"] == 299792458.0
    assert c_data["value"]["isExact"] is True
    assert "uncertainty" not in c_data["value"]

    # 2. Planck constant (exact, scientific notation preserved in valueDecimal)
    h_resp = client.get("/constants/PlanckConstant", headers={"Accept": "application/json"})
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["value"]["isExact"] is True
    assert math.isclose(h_data["value"]["valueFloat"], 6.62607015e-34, rel_tol=1e-15)
    assert not h_data["value"]["valueDecimal"].startswith("0.0000000000")
    assert h_data["value"]["nistExponent"] == -34

    # 3. Elementary charge, Boltzmann constant, Avogadro constant, Molar gas constant (all exact post-2019 SI)
    e_data = client.get("/constants/ElementaryCharge").json()
    k_data = client.get("/constants/BoltzmannConstant").json()
    na_data = client.get("/constants/AvogadroConstant").json()
    r_data = client.get("/constants/MolarGasConstant").json()

    assert e_data["value"]["isExact"] is True
    assert k_data["value"]["isExact"] is True
    assert na_data["value"]["isExact"] is True
    assert r_data["value"]["isExact"] is True
    assert math.isclose(
        r_data["value"]["valueFloat"],
        k_data["value"]["valueFloat"] * na_data["value"]["valueFloat"],
        rel_tol=1e-9,
    )

    # 4. Experimentally determined constants (FineStructureConstant, RydbergConstant)
    alpha_data = client.get("/constants/FineStructureConstant").json()
    rydberg_data = client.get("/constants/RydbergConstant").json()
    assert alpha_data["value"]["isExact"] is False
    assert float(alpha_data["value"]["uncertaintyFloat"]) > 0
    assert rydberg_data["value"]["isExact"] is False
    assert float(rydberg_data["value"]["uncertaintyFloat"]) > 0


def test_constant_version_endpoint(client):
    """Verify GET /constants/{id}/{versionId} returns specific historical adjustment value."""
    resp_1998 = client.get("/constants/PlanckConstant/1998")
    assert resp_1998.status_code == 200
    data_1998 = resp_1998.json()
    assert data_1998["versionId"] == "1998"
    assert data_1998["isExact"] is False
    assert "uncertainty" in data_1998


def test_content_negotiation_jsonld_html_turtle(client):
    """Verify JSON-LD @context in both application/ld+json and HTML embedded script."""
    # JSON-LD format
    ld_resp = client.get("/constants/PlanckConstant", headers={"Accept": "application/ld+json"})
    assert ld_resp.status_code == 200
    assert "application/ld+json" in ld_resp.headers["content-type"]
    ld_json = ld_resp.json()
    assert "@context" in ld_json

    # HTML format with embedded JSON-LD
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

    # Turtle format
    ttl_resp = client.get("/constants/PlanckConstant", headers={"Accept": "text/turtle"})
    assert ttl_resp.status_code == 200
    assert ttl_resp.headers["content-type"] == "text/turtle; charset=utf-8"
    assert "codata:nistExponent -34" in ttl_resp.text


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
    assert len(v2022.json().get("constants", [])) >= 350


def test_sparql_endpoint(client):
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
