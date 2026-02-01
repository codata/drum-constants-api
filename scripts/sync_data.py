import requests
from pathlib import Path
import sys

# Raw URL for the CODATA constants TTL file
SOURCE_URL = "https://raw.githubusercontent.com/codata/drum-constants/main/dist/rdf/codata_constants.ttl"
TARGET_FILE = Path(__file__).parent.parent / "src" / "data" / "codata_constants.ttl"

def sync_data():
    print(f"Syncing data from {SOURCE_URL}...")
    try:
        response = requests.get(SOURCE_URL, timeout=30)
        response.raise_for_status()
        
        # Ensure the target directory exists
        TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content to the file
        with open(TARGET_FILE, "wb") as f:
            f.write(response.content)
            
        print(f"Successfully updated {TARGET_FILE}")
        print(f"File size: {len(response.content) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error syncing data: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sync_data()
