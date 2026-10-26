from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
FIELDS_PATH = ROOT / "fields.yaml"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = ROOT / "report.md"
SUMMARY_FIELDS = ("kind", "priority")

CATEGORY_MAPPING = {
    "Basic Info": ["basic_info", "Basic Info"],
    "Technical Features": [
        "technical_features",
        "technical_characteristics",
        "Technical Features",
    ],
    "Performance Metrics": [
        "performance_metrics",
        "performance",
        "Performance Metrics",
    ],
    "Milestone Significance": [
        "milestone_significance",
        "milestones",
        "Milestone Significance",
    ],
    "Business Info": ["business_info", "commercial_info", "Business Info"],
    "Competition & Ecosystem": [
        "competition_ecosystem",
        "competition",
        "Competition & Ecosystem",
    ],
    "History": ["history", "History"],
    "Market Positioning": ["market_positioning", "market", "Market Positioning"],
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value.lower()).strip("-")
    return slug or "item"


def contains_uncertain(value: Any) -> bool:
    if isinstance(value, str):
        return "[uncertain]" in value
    if isinstance(value, list):
        return any(contains_uncertain(item) for item in value)
    if isinstance(value, dict):
        return any(contains_uncertain(item) for item in value.values())
    return False


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def category_aliases(category_name: str) -> list[str]:
    aliases = [category_name]
    aliases.extend(CATEGORY_MAPPING.get(category_name, []))
    normalized = re.sub(r"[^a-z0-9]+", "_", category_name.lower()).strip("_")
    aliases.append(normalized)
    return list(dict.fromkeys(aliases))


def find_in_nested(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        if field_name in value:
            return value[field_name]
        for child in value.values():
            found = find_in_nested(child, field_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_in_nested(child, field_name)
            if found is not None:
                return found
    return None


def lookup(data: dict[str, Any], category_name: str, field_name: str) -> Any:
    if field_name in data:
        return data[field_name]
    for alias in category_aliases(category_name):
        nested = data.get(alias)
        if isinstance(nested, dict) and field_name in nested:
            return nested[field_name]
    return find_in_nested(data, field_name)


def inline(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {inline(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(inline(item) for item in value)
    return str(value).replace("\n", " ")


def format_value(value: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value) and len(value) <= 3:
            return [prefix + ", ".join(str(item) for item in value)]
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                lines.append(prefix + "- " + " | ".join(
                    f"{key}: {inline(child)}" for key, child in item.items()
                ))
            elif isinstance(item, list):
                lines.append(prefix + "- " + inline(item))
            else:
                lines.append(prefix + "- " + str(item))
        return lines
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(prefix + f"- **{key}**")
                lines.extend(format_value(item, indent=indent + 1))
            else:
                lines.append(prefix + f"- **{key}:** {item}")
        return lines
    text = str(value)
    if len(text) > 100:
        return [prefix + "> " + part for part in text.splitlines()]
    return [prefix + text]


def defined_schema(fields_data: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], set[str]]:
    categories: list[tuple[str, list[dict[str, Any]]]] = []
    names: set[str] = set()
    for category_name, category in fields_data.get("categories", {}).items():
        fields = category.get("fields", [])
        categories.append((category_name, fields))
        names.update(field["name"] for field in fields)
    return categories, names


def top_level_extras(data: dict[str, Any], defined_names: set[str]) -> dict[str, Any]:
    nested_category_keys = {
        alias
        for aliases in CATEGORY_MAPPING.values()
        for alias in aliases
    }
    extras: dict[str, Any] = {}
    for key, value in data.items():
        if key in defined_names or key in {"_source_file", "uncertain"}:
            continue
        if key in nested_category_keys:
            continue
        extras[key] = value
    return extras


def main() -> None:
    fields_data = yaml.safe_load(FIELDS_PATH.read_text(encoding="utf-8"))
    categories, defined_names = defined_schema(fields_data)
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object: {path}")
        records.append((path, data))

    lines = [
        "# Deep-research field report",
        "",
        f"Generated from {len(records)} JSON result files. Values marked uncertain are omitted.",
        "",
        "## Table of contents",
        "",
    ]

    used_slugs: set[str] = set()
    toc_rows: list[tuple[str, str, dict[str, Any]]] = []
    for index, (path, data) in enumerate(records, start=1):
        title = str(data.get("name") or data.get("purpose") or path.stem.replace("_", " "))
        slug = slugify(title)
        base_slug = slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        used_slugs.add(slug)
        toc_rows.append((title, slug, data))
        summaries = []
        uncertain_fields = set(data.get("uncertain", []))
        for field_name in SUMMARY_FIELDS:
            value = lookup(data, "", field_name)
            if (
                field_name in uncertain_fields
                or is_empty(value)
                or contains_uncertain(value)
            ):
                continue
            summaries.append(f"{field_name}: {inline(value)}")
        suffix_text = f" - {' | '.join(summaries)}" if summaries else ""
        lines.append(f"{index}. [{title}](#{slug}){suffix_text}")

    for (path, data), (title, slug, _) in zip(records, toc_rows):
        lines.extend(["", f'<a id="{slug}"></a>', f"## {title}", ""])
        uncertain_fields = set(data.get("uncertain", []))
        for category_name, fields in categories:
            rendered: list[str] = []
            for field in fields:
                field_name = field["name"]
                value = lookup(data, category_name, field_name)
                if (
                    field_name in uncertain_fields
                    or is_empty(value)
                    or contains_uncertain(value)
                ):
                    continue
                rendered.extend([f"### {field_name}", ""])
                rendered.extend(format_value(value))
                rendered.append("")
            if rendered:
                lines.extend([f"### Category: {category_name}", ""])
                for line in rendered:
                    if line.startswith("### "):
                        lines.append("#### " + line[4:])
                    else:
                        lines.append(line)

        extras = top_level_extras(data, defined_names)
        if extras:
            lines.extend(["### Other Info", ""])
            for key, value in extras.items():
                if is_empty(value) or contains_uncertain(value):
                    continue
                lines.extend([f"#### {key}", ""])
                lines.extend(format_value(value))
                lines.append("")

        if uncertain_fields:
            lines.extend(["### Uncertain fields", ""])
            lines.extend(f"- {field_name}" for field_name in sorted(uncertain_fields))

    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} from {len(records)} result files")


if __name__ == "__main__":
    main()
