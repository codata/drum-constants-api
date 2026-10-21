import requests
from pathlib import Path
import sys

# Raw URLs for the CODATA constants TTL and ontology files
SOURCE_URLS = {
    "codata_constants.ttl": "https://raw.githubusercontent.com/codata/drum-constants/main/dist/rdf/codata_constants.ttl",
    "codata.ttl": "https://raw.githubusercontent.com/codata/drum-constants/main/dist/rdf/codata.ttl",
}
DATA_DIR = Path(__file__).parent.parent / "src" / "data"

def sync_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCE_URLS.items():
        target_file = DATA_DIR / filename
        print(f"Syncing data from {url}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(target_file, "wb") as f:
                f.write(response.content)
            print(f"Successfully updated {target_file} ({len(response.content) / 1024:.1f} KB)")
        except Exception as e:
            print(f"Error syncing {filename}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    sync_data()
