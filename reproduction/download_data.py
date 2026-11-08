"""Download the official LoCoMo release and record its immutable provenance."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FTEC5660-reproduction"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    directory = ROOT / "reproduction" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "locomo10.json"
    manifest_path = directory / "provenance.json"
    if destination.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(destination.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Dataset checksum mismatch; refusing to overwrite it")
        print("Verified existing dataset:", manifest["sha256"], flush=True)
        return
    commit = json.loads(fetch("https://api.github.com/repos/snap-research/locomo/commits/main"))["sha"]
    url = f"https://raw.githubusercontent.com/snap-research/locomo/{commit}/data/locomo10.json"
    raw = fetch(url)
    data = json.loads(raw)
    assert len(data) == 10 and all("qa" in sample and "conversation" in sample for sample in data)
    destination.write_bytes(raw)
    (directory / "LoCoMo-LICENSE.txt").write_bytes(fetch(
        f"https://raw.githubusercontent.com/snap-research/locomo/{commit}/LICENSE.txt"))
    manifest = {"repository": "https://github.com/snap-research/locomo", "commit": commit,
                "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "conversations": len(data), "questions": sum(len(s["qa"]) for s in data)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
