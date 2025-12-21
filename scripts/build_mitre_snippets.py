import json
from pathlib import Path

ATTACK_JSON = Path("data/raw/enterprise-attack.json")
OUTFILE = Path("data/mitre_techniques_snippets.json")


def make_short(desc: str, max_len: int = 220) -> str:
    """
    Naive truncation for a 'short summary' field.
    """
    desc = (desc or "").replace("\n", " ").strip()
    if len(desc) <= max_len:
        return desc
    return desc[:max_len].rsplit(" ", 1)[0] + " ..."


def main():
    bundle = json.loads(ATTACK_JSON.read_text(encoding="utf-8"))
    techniques = {}

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue

        tech_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id")
                break
        if not tech_id:
            continue

        name = obj.get("name", "")
        desc = obj.get("description", "")

        techniques[tech_id] = {
            "name": name,
            "short": make_short(desc),
        }

    OUTFILE.write_text(json.dumps(techniques, indent=2), encoding="utf-8")
    print(f"Wrote {len(techniques)} techniques to {OUTFILE}")


if __name__ == "__main__":
    main()