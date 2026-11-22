import argparse
import shutil
import sys
from pathlib import Path
import requests

# Raw URLs for the CODATA constants TTL and ontology files
SOURCE_URLS = {
    "codata_constants.ttl": "https://raw.githubusercontent.com/codata/drum-constants/main/dist/rdf/codata_constants.ttl",
    "codata.ttl": "https://raw.githubusercontent.com/codata/drum-constants/main/dist/rdf/codata.ttl",
}
DATA_DIR = Path(__file__).parent.parent / "src" / "data"
DEFAULT_LOCAL_DIR = Path(__file__).parent.parent.parent / "drum-constants" / "dist" / "rdf"


def sync_data(local_dir: Path | None = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if local_dir is not None:
        if not local_dir.exists():
            print(f"Error: local directory not found: {local_dir}", file=sys.stderr)
            sys.exit(1)
        for filename in SOURCE_URLS:
            src_file = local_dir / filename
            target_file = DATA_DIR / filename
            if not src_file.exists():
                print(f"Error: missing {src_file}", file=sys.stderr)
                sys.exit(1)
            shutil.copyfile(src_file, target_file)
            print(f"Synced {filename} from local {src_file} ({target_file.stat().st_size / 1024:.1f} KB)")
        return

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


def main():
    parser = argparse.ArgumentParser(description="Synchronize CODATA RDF data and ontology into src/data/")
    parser.add_argument(
        "--local",
        nargs="?",
        const=str(DEFAULT_LOCAL_DIR),
        default=None,
        help="Copy from local drum-constants dist/rdf directory instead of GitHub",
    )
    args = parser.parse_args()
    local_path = Path(args.local) if args.local else None
    sync_data(local_dir=local_path)


if __name__ == "__main__":
    main()
