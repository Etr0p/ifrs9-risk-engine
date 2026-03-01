#!/usr/bin/env python3
import json, sys, os

sys.stdout.reconfigure(encoding="utf-8")

JSONL_PATH = os.path.expanduser(os.path.join("~", ".claude", "projects", "C--tout-cours-programme", "58ae71be-99e0-4df3-bca7-3fd992bf7988.jsonl"))

SURVEY_KEYWORDS = ["research survey", "multi-asset", "vasicek", "eisenberg-noe", "vine copula", "hierarchical mas", "pma attribution", "shapley", "climate risk", "scope 3", "action plan", "roadmap", "specifications", "research summary", "research synthesis", "research report", "paysage acad", "plan strat", "assetclassprofile", "multi-actifs", "ngfs", "scope3"]

PRIORITY_LINES = {5197, 5201, 5203, 5205, 5207, 5209, 5211, 5215, 5218, 5220, 7000, 7003, 7004, 7007, 7008, 7011, 7012, 7015, 7017, 7020, 7021, 7024, 7025, 7028, 7029}

def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return chr(10).join(parts)
    return ""

def extract_task_result(text):
    if "<result>" in text and "</result>" in text:
        start = text.index("<result>") + len("<result>")
        end = text.index("</result>")
        return text[start:end].strip()
    return text

def matches_keywords(text):
    lower = text.lower()
    return any(kw in lower for kw in SURVEY_KEYWORDS)

def print_section(line_no, role, text, label=None):
    if label:
        print()
        print("#" * 120)
        print(f"# {label}")
        print("#" * 120)
    sep = "-" * 100
    print(f"{chr(10)}{sep}")
    print(f"| LINE {line_no} | {role.upper()} | {len(text)} chars")
    print(f"{sep}{chr(10)}")
    print(text)
    print()

def main():
    sections = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if obj.get("type") not in ("user", "assistant"): continue
            role = obj["type"]
            msg = obj.get("message", {})
            raw_text = extract_text(msg.get("content", ""))
            if not raw_text.strip(): continue
            is_priority = i in PRIORITY_LINES
            is_keyword_match = matches_keywords(raw_text) and len(raw_text) > 1000
            if is_priority or is_keyword_match:
                text = extract_task_result(raw_text)
                sections.append((i, role, text))

    print("=" * 120)
    print("EXTRACTED RESEARCH SURVEY, ACTION PLANS & TECHNICAL SPECIFICATIONS")
    print(f"Source: {JSONL_PATH}")
    print(f"Total sections extracted: {len(sections)}")
    print("=" * 120)

    groups = [
        ("PART A -- VIRTUAL CRO RESEARCH SURVEYS (Sub-Agent Results)", {5197, 5201, 5203, 5205, 5207, 5209, 5211}),
        ("PART B -- SYNTHESIS & COUNTER-ANALYSIS", {5215, 5218, 5220}),
        ("PART C -- MULTI-ASSET RESEARCH SURVEY & LANDSCAPE", {7000, 7003}),
        ("PART D -- STRATEGIC PLAN, ACTION PLAN & TECHNICAL SPECS", {7004, 7007, 7008, 7011, 7012}),
        ("PART E -- COUNTER-ANALYSES, REFINEMENTS & DOC", {7015, 7017, 7020, 7021, 7024, 7025, 7028, 7029}),
    ]

    all_grouped = set()
    for _, line_set in groups:
        all_grouped |= line_set

    for group_label, line_set in groups:
        first = True
        for line_no, role, text in sections:
            if line_no in line_set:
                label = group_label if first else None
                first = False
                print_section(line_no, role, text, label)

    remaining = [(ln, r, t) for ln, r, t in sections if ln not in all_grouped]
    if remaining:
        print()
        print("#" * 120)
        print("# ADDITIONAL KEYWORD-MATCHING SECTIONS")
        print("#" * 120)
        for line_no, role, text in remaining:
            print_section(line_no, role, text)

    print()
    print("=" * 120)
    print("EXTRACTION SUMMARY")
    print("=" * 120)
    total_chars = sum(len(t) for _, _, t in sections)
    print(f"Total sections: {len(sections)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated pages (2500 chars/page): {total_chars / 2500:.0f}")
    print()
    print("Sections by part:")
    for group_label, line_set in groups:
        count = sum(1 for ln, _, _ in sections if ln in line_set)
        chars = sum(len(t) for ln, _, t in sections if ln in line_set)
        short = group_label.split(" -- ")[1].strip() if " -- " in group_label else group_label
        print(f"  {short}: {count} sections, {chars:,} chars")
    if remaining:
        chars = sum(len(t) for _, _, t in remaining)
        print(f"  Additional matches: {len(remaining)} sections, {chars:,} chars")

if __name__ == "__main__":
    main()
