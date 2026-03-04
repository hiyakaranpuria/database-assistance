"""
generate_modelfile.py — Dynamically regenerate the Modelfile with the user's actual DB schema.

Called by entrypoint.py after database connection is established.
Usage:  python generate_modelfile.py
"""

import os
import re
from metadata_provider import extract_metadata

MODELFILE_PATH = os.path.join(os.path.dirname(__file__), "Modelfile")


def build_schema_block():
    """Build the ### DATABASE SCHEMA section from live metadata."""
    metadata = extract_metadata()
    if not metadata:
        return "### DATABASE SCHEMA\n(No collections found — connect to a database first)\n"

    lines = ["### DATABASE SCHEMA"]
    for coll_name, info in sorted(metadata.items()):
        fields = info.get("fields", {})
        field_strs = [f"{fname}({ftype})" for fname, ftype in fields.items()]
        lines.append(f"Collection '{coll_name}': {{{', '.join(field_strs)}}}")

    # Build join hints automatically
    # Look for fields ending in 'Id' to suggest relationships
    join_hints = []
    for coll_name, info in sorted(metadata.items()):
        for fname in info.get("fields", {}):
            if fname.endswith("Id") and fname != "_id":
                target = fname[:-2] + "s"  # e.g. customerId -> customers
                if target in metadata:
                    join_hints.append(
                        f"- '{coll_name}' links to '{target}' via '{fname}'."
                    )

    if join_hints:
        lines.append("")
        lines.append("### JOIN RELATIONSHIPS (auto-detected)")
        lines.extend(join_hints)

    return "\n".join(lines)


def regenerate_modelfile():
    """Read the existing Modelfile, replace the schema section, write it back."""
    with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    new_schema = build_schema_block()

    # Replace everything between "### DATABASE SCHEMA" and the next "###" or "EXAMPLE:"
    pattern = r"### DATABASE SCHEMA.*?(?=###\s+[^\n]*RULES|EXAMPLE:)"
    replacement = new_schema + "\n\n"
    updated = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"✅ Modelfile updated with schema from database '{os.getenv('MONGO_DB', 'unknown')}'")
    print(f"   Collections found: {len(extract_metadata())}")


if __name__ == "__main__":
    regenerate_modelfile()
