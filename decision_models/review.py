"""Create blinded offline review forms and import independent human annotations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from label import RUBRIC
from prepare import family, sample
from schemas import body_hash, read_records, save, write_records


FORM = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Financial document review</title>
<style>body{max-width:850px;margin:35px auto;padding:0 20px;font:17px/1.6 system-ui;color:#182332}
pre{white-space:pre-wrap;font:inherit}button,select{font:inherit;margin:8px;padding:8px}
textarea{width:100%;font:inherit}aside{padding:16px;background:#edf3fa}#body{border-top:1px solid #ddd;padding-top:20px}</style>
<h1>Independent document review</h1><aside>Use the supplied document alone. Routine: ordinary information
without an apparent development warranting closer review. Review-worthy: a potentially significant
corporate, financial, economic or policy development. Unclear: insufficient or conflicting content.
Scheduled earnings or policy releases can still warrant review. Allow at most three hours total;
leave unfinished documents unreviewed. Your answers are not investment recommendations.</aside>
<p id="progress"></p><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button>
<select id="label" onchange="remember()"><option value="">Unreviewed</option><option>routine</option>
<option>review_worthy</option><option>unclear</option></select>
<textarea id="reason" rows="2" placeholder="Optional short reason or uncertainty" oninput="remember()"></textarea>
<button onclick="download()">Save annotations</button><pre id="body"></pre>
<script>
const rows=__ROWS__, key='triage-review-'+__PACK__, answers=JSON.parse(localStorage.getItem(key)||'{}');let index=0;
function show(){const r=rows[index],a=answers[r.review_id]||{};
document.getElementById('progress').textContent=`Document ${index+1}/${rows.length} · ${Object.keys(answers).filter(k=>answers[k].label).length} reviewed`;
document.getElementById('body').textContent=r.body;document.getElementById('label').value=a.label||'';
document.getElementById('reason').value=a.reason||'';}
function remember(){answers[rows[index].review_id]={label:document.getElementById('label').value,reason:document.getElementById('reason').value};
localStorage.setItem(key,JSON.stringify(answers));}
function move(n){remember();index=Math.min(rows.length-1,Math.max(0,index+n));show();}
function download(){remember();const out=rows.filter(r=>answers[r.review_id]?.label).map(r=>({review_id:r.review_id,...answers[r.review_id]}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
a.download=key+'.json';a.click();URL.revokeObjectURL(a.href);}
show();</script></html>'''


def build(splits, output):
    rows = read_records(splits / "test-input.jsonl")
    train = read_records(splits / "train-input.jsonl")
    by_family = {}
    for row in sample(rows):
        by_family.setdefault(family(row), []).append(row)
    selected = [row for name in sorted(by_family) for row in by_family[name][:12]]
    if not selected:
        raise ValueError("No fresh test documents")
    output.mkdir(parents=True, exist_ok=True)
    identities, packs = {}, {}
    first, second = selected[::2], selected[1::2]
    # Twelve independently double-reviewed documents; identities do not reveal overlap to reviewers.
    overlap = [row for name in sorted(by_family) for row in by_family[name][:2]]
    packs["reviewer-a"] = first + [r for r in second if r in overlap]
    packs["reviewer-b"] = second + [r for r in first if r in overlap]
    packs["rubric-pilot"] = []
    for name in sorted(by_family):
        packs["rubric-pilot"].extend([r for r in sample(train) if family(r) == name][:2])
    for name, documents in packs.items():
        blind = []
        for i, row in enumerate(documents):
            review_id = f"{name}-{i + 1:03}"
            identities[review_id] = {"body_sha256": body_hash(row), "pack": name,
                                     "purpose": "rubric" if name == "rubric-pilot" else "evaluation"}
            blind.append({"review_id": review_id, "body": row["body"]})
        identifier = hashlib.sha256(json.dumps(blind).encode()).hexdigest()[:12]
        content = FORM.replace("__ROWS__", json.dumps(blind).replace("<", "\\u003c"))
        content = content.replace("__PACK__", json.dumps(identifier))
        (output / f"{name}.html").write_text(content, encoding="utf-8")
    save({"mapping": identities, "test_families": dict(Counter(family(r) for r in selected)),
          "independent_test_documents": len(selected), "rubric": RUBRIC, "hours_total": 3}, output / "mapping.json")
    print(f"Prepared {len(selected)} independent documents; overlap up to 12; blind forms in {output}")


def import_reviews(mapping, reviews, dataset, output):
    mapping = json.loads(mapping.read_text(encoding="utf-8"))["mapping"]
    annotations = {}
    for path in reviews:
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = row["review_id"]
            if key not in mapping or row["label"] not in ("routine", "review_worthy", "unclear"):
                raise ValueError("Unknown review identity or label")
            identity = mapping[key]
            if identity["purpose"] == "evaluation":
                votes = annotations.setdefault(identity["body_sha256"], {})
                if identity["pack"] in votes and votes[identity["pack"]] != row:
                    raise ValueError("Conflicting exports from the same reviewer")
                votes[identity["pack"]] = row
    rows = read_records(dataset)
    agreed = unresolved = overlaps = matching = 0
    for row in rows:
        votes = list(annotations.get(body_hash(row), {}).values())
        if not votes:
            continue
        labels = {r["label"] for r in votes}
        overlaps += len(votes) > 1
        matching += len(votes) > 1 and len(labels) == 1
        if len(labels) == 1:
            row["human"] = {"label": votes[0]["label"], "uncertainty": None, "reviews": votes}
            agreed += 1
        else:
            row["human"] = {"label": None, "uncertainty": None, "reviews": votes, "unresolved": True}
            unresolved += 1
    write_records(rows, output)
    save({"reviewed": agreed + unresolved, "resolved": agreed, "unresolved": unresolved,
          "overlapping": overlaps, "matching": matching,
          "agreement": matching / overlaps if overlaps else None}, output.with_suffix(".agreement.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--reviews", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if args.reviews:
        if not args.mapping or not args.dataset:
            parser.error("Import requires --mapping and --dataset")
        import_reviews(args.mapping, args.reviews, args.dataset, args.output)
    elif args.splits:
        build(args.splits, args.output)
    else:
        parser.error("Supply --splits to build a pack or --reviews to import it")
