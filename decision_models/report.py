"""Build a source-backed Markdown/PDF research report from verified frozen test artifacts."""

from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
from xml.sax.saxutils import escape

import numpy as np

from evaluate_study import bootstrap, validate_freeze
from protocol import breakdown, fingerprint
from schemas import HERE, LABELS, body_hash, load_split, read_records, save


def number(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def ci(value):
    return "-" if value["lower"] is None else f"[{value['lower']:.3f}, {value['upper']:.3f}]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def label(config):
    kind = {"laya": "Laya", "tfidf": "Word TF-IDF", "char": "Char TF-IDF", "combined": "Combined TF-IDF",
            "majority": "Majority", "finbert": "FinBERT", "bge": "BGE", "modernbert": "ModernBERT",
            "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B"}[config.get("kind", "laya")]
    suffix = config.get("classifier", config.get("adaptation", "SDK"))
    if config["family"] == "research_zero":
        suffix += " epoch zero"
    return kind + " " + suffix


def human_audit(path, records, predictions, reference):
    if path is None:
        return {"status": "Pending blinded non-expert review", "resolved": 0, "unresolved": 0}
    reviewed = read_records(path)
    if {body_hash(r) for r in reviewed} != {body_hash(r) for r in records} or len(reviewed) != len(records):
        raise ValueError("Human audit must overlay exactly the frozen test bodies")
    labels = {body_hash(r): r["human"]["label"] for r in reviewed if r["human"]["label"] is not None}
    unresolved = sum(bool(r["human"].get("unresolved")) for r in reviewed)
    subset = {name: [{**r, "reference": labels[r["body_sha256"]], "reference_source": "human"}
                     for r in rows if r["body_sha256"] in labels] for name, rows in predictions.items()}
    truth = {body_hash(r): r["llm"]["label"] for r in records}
    return {"status": "Blinded non-expert audit; stratified review sample, not analyst ground truth",
            "source_sha256": fingerprint(path), "resolved": len(labels), "unresolved": unresolved,
            "teacher_agreement": sum(truth[k] == value for k, value in labels.items()) / len(labels) if labels else None,
            "models": {name: breakdown(rows) for name, rows in subset.items()},
            "bootstrap": bootstrap(subset, reference) if labels else None}


def corpus_tables(prepared):
    parts = []
    if prepared.get("census"):
        parts += [table(["Input", "Raw documents", "Family / provider", "Document dates"],
                        [[Path(row["input"]).name + (" (legacy)" if row["legacy"] else ""), row["rows"],
                          ", ".join(row["families"]) + " / " + ", ".join(row["providers"]),
                          f"{row['start']} to {row['end']}"] for row in prepared["census"]]),
                  "Raw acquisition counts precede duplicate quarantine, sampling and labeling. Legacy inputs were previously "
                  "experimented on and are eligible only for training. SEC dates are filing dates, including release exhibits, "
                  "and may be later than publication elsewhere. Other sources use original article dates or publication metadata; "
                  "temporal splits use these recorded document dates at day precision."]
    if prepared.get("legacy_extraction_check"):
        legacy = prepared["legacy_extraction_check"]
        parts += [table(["Legacy text formatter", "Primary filings reconstructed", "Exact old-body matches"],
                        [[f"edgartools {legacy['sdk_version']}", legacy["attempted"], legacy["matched_documents"]]]),
                  "Offline reconstruction of retained primary HTML links exact old-body hashes across text-extraction formats. "
                  "These links also quarantine related release exhibits from fresh evaluation. Matching candidates are not "
                  "additional exclusion counts; their groups enter the exclusion rules below. Canonical model inputs are unchanged."]
    if prepared.get("exclusions_by_family"):
        parts += [table(["Family", "Exclusion reason", "Documents"],
                        [[name, reason.replace("_", " "), count]
                         for name, counts in prepared["exclusions_by_family"].items() for reason, count in counts.items()])]
    if prepared.get("partitions"):
        parts += [table(["Partition", "Available after exclusions", "Selected before labeling"],
                        [[name, counts["available"], counts["chosen"]] for name, counts in prepared["partitions"].items()]),
                  "Selection applies the recorded split ceilings and train/validation/test 8-K share cap; these counts are distinct "
                  "from labeled support below."]
    return parts


def reference_quote(records):
    """Local cost scenarios only: no API client, credentials, dispatch or paid benchmark."""
    import tiktoken
    from label import RUBRIC
    tokenizer = tiktoken.get_encoding("o200k_base")
    estimated = [len(tokenizer.encode(RUBRIC + r["body"], disallowed_special=())) + 128 for r in records]
    bound = [len((RUBRIC + r["body"]).encode("utf-8")) + 2048 for r in records]
    scenarios = {}
    for output in (128, 1024):
        scenarios[str(output)] = {"approximate_usd": sum(n / 1e6 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in estimated),
                                 "conservative_usd": sum(n / 1e6 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in bound)}
    return {"model": "gpt-6.1-sol", "tier": "flex", "executed": False, "documents": len(records),
            "rates_usd_per_million": {"input": 1.0, "cached_input": .05, "output": 5.0},
            "pricing_checked": "2026-10-09", "pricing_source": "https://developers.openai.com/api/docs/pricing/",
            "tokenizer": "o200k_base approximation; UTF-8 byte upper bound plus 2048 prompt/schema overhead",
            "scenarios_output_tokens_per_document": scenarios,
            "limits": "No cache discounts. Above 272k input, 2x input/1.5x output. Reasoning counts as output; not a service guarantee."}


def cached_weights(config):
    if not config.get("model"):
        return None
    from huggingface_hub import try_to_load_from_cache
    from pathlib import Path
    configuration = try_to_load_from_cache(config["model"], "config.json", revision=config["revision"])
    if not isinstance(configuration, str):
        return None
    snapshot = Path(configuration).parent
    sizes = [p.stat().st_size for p in snapshot.rglob("*") if p.suffix in (".bin", ".pt", ".safetensors")]
    return sum(sizes) if sizes else None


def figures(output, entries, configs, results, intervals, main, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    grouped = defaultdict(list)
    for name, entry in entries.items():
        if entry["role"] == "curve":
            grouped[label(configs[name])].append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for title, names in grouped.items():
        names.sort(key=lambda n: entries[n]["samples"])
        x = [entries[n]["samples"] for n in names]
        y = [results[n]["macro_f1"] for n in names]
        ax.plot(x, y, "o-", label=title, alpha=.8)
    ax.set(xscale="log", xlabel="Training documents (fixed recipe, additional tuning labels)", ylabel="Test macro F1",
           ylim=(-.02, 1.05), title="Sample efficiency on the same temporal test cohort")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=180)
    plt.close(fig)
    names = [n for n, e in entries.items() if e["kind"] == "laya" and e["role"] in ("context", "curve") and
             e["samples"] == min(max(e["samples"] for e in entries.values()), 4000)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for adaptation in ("head", "lora"):
        selected = sorted([n for n in names if configs[n]["adaptation"] == adaptation], key=lambda n: configs[n]["max_len"])
        x, y = [configs[n]["max_len"] for n in selected], [results[n]["macro_f1"] for n in selected]
        if selected:
            line, = ax.plot(x, y, "o-", label=adaptation)
            ax.vlines(x, [intervals[n]["macro_f1"]["lower"] for n in selected],
                      [intervals[n]["macro_f1"]["upper"] for n in selected], color=line.get_color())
    ax.set(xlabel="Laya total context tokens", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Context ablation; paired group-bootstrap 95% intervals")
    if names:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "context.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in main:
        perf = json.loads((output.parent / entries[name]["directory"] / "performance.json").read_text())
        x = perf["batch_throughput"]["1"]["documents_per_second"]
        ax.scatter(x, results[name]["macro_f1"])
        ax.annotate(label(configs[name]), (x, results[name]["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xscale="log", xlabel="Warm single-document throughput (documents/s)", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Agreement and local throughput; shared RTX 3060/CPU environment")
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, primary, strict=True):
        matrix = np.array(results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["routine", "review", "unclear"],
               yticklabels=["routine", "review", "unclear"], xlabel="Prediction", ylabel="Teacher reference", title=label(configs[name]))
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=180)
    plt.close(fig)


def pdf(markdown, destination):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize, styles["BodyText"].leading = 10, 14
    cell = ParagraphStyle("Cell", fontSize=7.5, leading=10, alignment=TA_LEFT)
    header = ParagraphStyle("Header", parent=cell, textColor=colors.white)
    width = A4[0] - 84
    flow = []
    def text(value):
        value = escape(value).replace("`", "")
        value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1855a0">\1</link>', value)
        return value
    lines, index = markdown.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("| "):
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                columns = [value.strip() for value in lines[index].strip("| ").split("|")]
                if not all(value == "---" for value in columns):
                    rows.append([Paragraph(text(value), header if not rows else cell) for value in columns])
                index += 1
            widths = [width / len(rows[0])] * len(rows[0])
            if len(widths) > 4:
                widths = [width * .28] + [width * .72 / (len(widths) - 1)] * (len(widths) - 1)
            content = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            content.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203b58")),
                                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")])]))
            flow += [KeepTogether([content, Spacer(1, 12)])] if len(rows) <= 6 else [content, Spacer(1, 12)]
            continue
        match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if match:
            image = Image(str(destination.parent / match[2]))
            image.drawHeight = image.imageHeight * width / image.imageWidth
            image.drawWidth = width
            flow += [image, Paragraph(text(match[1]), styles["Caption"] if "Caption" in styles else cell), Spacer(1, 12)]
        elif line:
            level = len(line) - len(line.lstrip("#"))
            style = styles["Title" if level == 1 else "Heading2" if level else "BodyText"]
            flow.append(Paragraph(text(line.lstrip("# ") if level else line), style))
        index += 1
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 25, "Financial-document triage | fixed-seed research study")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))
    SimpleDocTemplate(str(destination), title="Financial-document triage across public sources",
                      author="Financial-document triage research", pagesize=A4,
                      rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42).build(
        flow, onFirstPage=footer, onLaterPages=footer)


def build(splits, study, human=None):
    freeze = validate_freeze(splits, study)
    results = json.loads((study / "results.json").read_text(encoding="utf-8"))
    for key, name in (("bootstrap", "bootstrap.json"), ("matched_coverage", "matched-coverage.json")):
        if fingerprint(study / name) != results[f"{key}_sha256"]:
            raise ValueError("Report statistics changed after test summarization")
    entries = freeze["entries"]
    predictions, configs, scored = {}, {}, {}
    for name, entry in entries.items():
        path = study / "test" / f"{name}.json"
        if fingerprint(path) != results["test_artifacts"][name]:
            raise ValueError("Test evidence changed before report generation")
        test = json.loads(path.read_text(encoding="utf-8"))
        if test["freeze_sha256"] != fingerprint(study / "freeze.json"):
            raise ValueError("Report test results use another freeze")
        predictions[name], scored[name] = test["predictions"], test["metrics"]["overall"]
        configs[name] = json.loads((study / entry["directory"] / "config.json").read_text(encoding="utf-8"))
    records = load_split(splits, "test")
    audit = human_audit(human, records, predictions, freeze["primary_linear"])
    quote = reference_quote(records)
    output = study / "report"
    output.mkdir(exist_ok=True)
    save(audit, output / "human-audit.json")
    save(quote, output / "reference-quote.json")
    intervals = json.loads((study / "bootstrap.json").read_text())["models"]
    main = {}
    for name, entry in entries.items():
        if entry["role"] in ("curve", "reference"):
            key = label(configs[name])
            if key not in main or entry["samples"] > entries[main[key]]["samples"]:
                main[key] = name
    names = list(main.values())
    raw_scores = {name: json.loads((study / "test" / f"{name}.json").read_text())["raw_metrics"]["overall"] for name in names}
    primary, reference = freeze["primary_decision"], freeze["primary_linear"]
    figures(output, entries, configs, scored, intervals, names, (primary, reference))
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    delta = intervals[primary]["paired_delta_vs_reference"]["macro_f1"]
    conclusion = ("Adapted Laya improves teacher agreement over the selected linear baseline on this test cohort."
                  if delta["lower"] > 0 else "Adapted Laya agrees less well with the teacher than the selected linear baseline."
                  if delta["upper"] < 0 else "This test does not resolve a macro-F1 advantage for adapted Laya over the selected linear baseline.")
    parts = ["# Financial-document triage across public sources", "", conclusion,
             f"The primary validation-selected contrast is {label(configs[primary])} versus {label(configs[reference])}. "
             f"Test macro F1 is {scored[primary]['macro_f1']:.3f} versus {scored[reference]['macro_f1']:.3f}; "
             f"paired delta 95% interval {ci(delta)}. These are agreement scores against Luna labels.",
             f"Independent audit: {audit['status']}; {audit['resolved']} resolved and {audit['unresolved']} unresolved documents. "
             "No claim of analyst usefulness or investment profitability follows from teacher agreement alone.",
             "", "## Task and evaluation", "",
             "Classify supplied source content as routine, review_worthy or unclear for an analyst inbox. "
             "No issuer history, portfolio context, future returns or market reaction is supplied. Unclear routes to review. "
             "Macro F1 averages all three declared labels, including absent classes in small slices. "
             "The critical error is a review-worthy document predicted routine.",
             "The research question is whether supervised decision-model adaptation and larger contiguous context improve "
             "triage agreement, critical misses and throughput relative to lexical, encoder and small decoder baselines. "
             "FinBERT's published sentiment results do not establish triage quality; BGE is an embedding model. "
             "ModernBERT controls for Laya's backbone, while Qwen tests a bounded non-thinking deployment. "
             f"Primary sources and distinctions are documented in [literature.md](https://github.com/LuccaPereiraMartins/systematic-trading/blob/{git_revision}/decision_models/literature.md).",
             "Train ends 2026-03-31; selection/calibration use April-June; test uses July-September. "
             "Event and verified near-duplicate groups cannot cross boundaries. Old experimental bodies and near duplicates "
             "are excluded from fresh validation/test. Candidate retrieval is approximate and can miss duplicates. "
             "Every core recipe, checkpoint, calibration and discard threshold was frozen before test inference.",
             "", "## Source and label coverage", ""]
    if "synthetic" in freeze["manifest"]["prepared"].get("method", "").lower():
        parts.insert(1, "Synthetic engineering fixture: this document verifies the pipeline and contains no research findings.")
    manifest = freeze["manifest"]["splits"]
    parts += corpus_tables(freeze["manifest"]["prepared"])
    families = sorted(set.union(*[set(v.get("families", {})) for v in manifest.values()]))
    parts += [table(["Family", "Train", "Selection", "Calibration", "Test"],
                    [[f, *[manifest[n].get("families", {}).get(f, 0) for n in ("train", "selection", "calibration", "test")]] for f in families]),
              "Source shortfalls are not filled by additional 8-Ks. These institutional/public-news cohorts are not a representative "
              "commercial inbox. Publication dates and snapshots cannot establish the exact text available at the historical date; "
              "pretraining exposure remains possible. Rights and collection methods are documented in "
              f"[sources.md](https://github.com/LuccaPereiraMartins/systematic-trading/blob/{git_revision}/decision_models/sources.md).",
              table(["Split", "Documents", *LABELS], [[n, manifest[n]["count"], *[manifest[n].get("labels", {}).get(k, 0) for k in LABELS]]
                                                     for n in ("train", "selection", "calibration", "test")]),
              "", "## Model comparisons", "",
              table(["Model", "Train", "Macro F1", "95% CI", "Misses", "Miss rate", "Brier"],
                    [[label(configs[n]), entries[n]["samples"], number(scored[n]["macro_f1"]), ci(intervals[n]["macro_f1"]),
                      scored[n]["default_discard"]["dangerous_misses"], number(scored[n]["default_discard"]["dangerous_miss_rate"]),
                      number(scored[n]["brier"])] for n in names]),
              "Misses use the default argmax routine decision; the denominator is review-worthy support. "
              "Brier sums squared errors over three classes (0-2). Full artifacts include class precision/recall/F1, confusion matrices, "
              "accuracy, log loss, ten-bin ECE, source slices and reference provenance. Intervals resample whole related groups "
              "2000 times with shared draws; they condition on the single trained seed and have no multiplicity correction.",
              "![Confusion matrices show the critical review-to-routine cell](confusion.png)",
              table(["Model", "Class", "Precision", "Recall", "F1", "Support"],
                    [[label(configs[n]), name, number(value['precision']), number(value['recall']), number(value['f1']), value['support']]
                     for n in (primary, reference) for name, value in scored[n]['per_class'].items()]),
              "", "## Source slices and uncertainty", "",
              table(["Model", "Family", "Documents", "Macro F1", "Review support", "Misses"],
                    [[label(configs[n]), family, value['count'], number(value['macro_f1']),
                      value['default_discard']['review_worthy_support'], value['default_discard']['dangerous_misses']]
                     for n in (primary, reference) for family, value in breakdown(predictions[n])['by_family'].items()]),
              table(["Model", "Provider", "Documents", "Macro F1", "Review support", "Misses"],
                    [[label(configs[n]), source, value['count'], number(value['macro_f1']),
                      value['default_discard']['review_worthy_support'], value['default_discard']['dangerous_misses']]
                     for n in (primary, reference) for source, value in breakdown(predictions[n])['by_source'].items()]),
              table(["Model", "Log loss", "ECE (10 bins)", "Median uncertainty"],
                    [[label(configs[n]), number(scored[n]['log_loss']), number(scored[n]['ece_10_bins']),
                      number(float(np.median([r['uncertainty'] for r in predictions[n]])))] for n in names]),
              table(["Model", "Raw Brier", "Cal. Brier", "Raw ECE", "Cal. ECE"],
                    [[label(configs[n]), number(raw_scores[n]['brier']), number(scored[n]['brier']),
                      number(raw_scores[n]['ece_10_bins']), number(scored[n]['ece_10_bins'])] for n in names]),
              "Raw and calibrated scores use the same test documents. A scalar temperature is fitted on separate "
              "calibration groups; it changes confidence and discard behavior while preserving argmax labels. "
              "Its use follows [Guo et al.](https://arxiv.org/abs/1706.04599); it does not guarantee calibration on this cohort.",
              "Uncertainty is 1 minus the largest calibrated class probability. It is model confidence, not a probability "
              "of investment materiality; teacher self-reported uncertainty is retained separately. Class support must be read "
              "beside small-slice macro F1. When no misses are observed, bootstrap resampling cannot reveal unseen errors.",
              "", "## Discard policies", ""]
    source_counts = {}
    for split in ("train", "selection", "calibration", "test"):
        counts = defaultdict(int)
        for row in load_split(splits, split):
            counts[row.get("source", "sec")] += 1
        source_counts[split] = counts
    providers = sorted(set.union(*[set(counts) for counts in source_counts.values()]))
    parts.insert(parts.index("## Model comparisons") - 1,
                 table(["Provider", "Train", "Selection", "Calibration", "Test"],
                       [[source, *[source_counts[split][source] for split in source_counts]] for source in providers]) +
                 "\n\nThe news family can contain official financial announcements as well as Wikinews. "
                 "Provider coverage is reported separately; official announcements do not establish commercial-journalism generalization.")
    coverage = freeze["manifest"].get("label_coverage", {})
    if coverage:
        parts.insert(parts.index("## Model comparisons") - 1,
                     table(["Label queue", "Queued", "Labeled", "Unlabeled", "Retained", "8-K cap excluded"],
                           [[name, value['queued'], value['labeled'], value['unlabeled'],
                             value.get('retained', value['labeled']), value.get('excluded_8k_share', 0)]
                            for name, value in coverage.items()]) +
                     "\n\nValidation/test labeling must be complete. A budget-limited training queue can retain unlabeled rows; "
                     "family/error counts remain in the manifest. The 40% primary 8-K cap is checked again on retained labels; "
                     "cap exclusions remain in the input queue and project spending. Attained sample sizes, rather than collection targets, define the learning curves.")
        if all("teacher_cost_usd" in value for value in coverage.values()):
            parts.insert(parts.index("## Model comparisons") - 1,
                         table(["Label queue", "Successful teacher labels", "Recorded USD", "Cost unavailable"],
                               [[name, value['successful_teacher_labels'], number(value['teacher_cost_usd'], 4),
                                 value['teacher_cost_unavailable']] for name, value in coverage.items()]) +
                         "\n\nThese costs cover successful teacher annotations retained in the frozen benchmark, "
                         "using recorded API usage and prices. Missing costs are counted separately. The shared $3 project ledger "
                         "also covers pilots, failed or unresolved requests and labels excluded from the benchmark; "
                         "these retained-label totals are not the entire project bill. Human-review time is excluded.")
    policy_rows = []
    for n in names:
        for key, value in scored[n]["policies"].items():
            policy_rows.append([label(configs[n]), key, number(value["workload_reduction"]), value["dangerous_misses"],
                                value["review_worthy_support"], number(value["dangerous_miss_rate"]), number(value["discard_contamination"])])
    parts += [table(["Model", "Target", "Workload reduction", "Misses", "Support", "Miss rate", "Contamination"], policy_rows),
              "Thresholds maximize discarded routine predictions subject to empirical calibration miss targets of 1%/5%. "
              "Predicted review-worthy/unclear always escalate. Sparse support and zero observed misses do not establish "
              "population guarantees. Test rates, rather than calibration targets, describe the observed deployment tradeoff.",
              "", "## Adaptation, sample efficiency and context", "",
              "![Nested learning curves on the same held-out cohort](learning-curves.png)",
              "Training sizes are nested 250/1000/4000/16000/all where attainable. Each neural run restarts from pinned pretrained "
              "weights; one training seed is used. Recipes are screened at 1000/4000 or nearest attainable sizes. Their tuning labels "
              "are additional, so the smallest point is not a claim that its training rows alone suffice to discover the recipe.",
              "![Context ablations retain the same whole documents](context.png)",
              "Laya uses fixed uniform whole-document pooling and 50% overlapping windows; the native SDK reference uses "
              "most-confident-window aggregation. CE/Brier objectives are document-level; no chunk labels are invented. "
              "Matched pre-update controls separate supervised adaptation from aggregation and k-bit precision changes. "
              "FinBERT/BGE/ModernBERT use fresh triage heads; FinBERT's sentiment head is not mapped to triage labels. "
              "Qwen uses non-thinking NF4/BF16 A/B/C scoring; QLoRA is conditional discriminative CE, not full-vocabulary SFT."]
    contrasts = json.loads((study / "bootstrap.json").read_text())["contrasts"]
    parts += [table(["Adapted model", "Train", "Matched-zero F1", "Adapted F1", "Paired gain 95% CI"],
                    [[label(configs[n]), entries[n]['samples'], number(scored[entries[n]['matched_zero']]['macro_f1']),
                      number(scored[n]['macro_f1']), ci(contrasts[n + '-adaptation']['metrics']['macro_f1'])]
                     for n in names if entries[n].get('matched_zero')]),
              table(["Model", "Training increment", "F1 gain", "Paired gain 95% CI"],
                    [[label(configs[v['left']]), f"{entries[v['right']]['samples']} to {entries[v['left']]['samples']}",
                      number(scored[v['left']]['macro_f1'] - scored[v['right']]['macro_f1']), ci(v['metrics']['macro_f1'])]
                     for k, v in contrasts.items() if k.endswith('-sample-increment')]),
              table(["Model", "Context tokens", "F1 gain", "Paired gain 95% CI"],
                    [[label(configs[v['left']]), f"{configs[v['right']]['max_len']} to {configs[v['left']]['max_len']}",
                      number(scored[v['left']]['macro_f1'] - scored[v['right']]['macro_f1']), ci(v['metrics']['macro_f1'])]
                     for k, v in contrasts.items() if k.endswith('-context-increment')]),
              "Wider Laya windows change contiguous context and the number of votes under the same pooling rule. "
              "All source tokens remain covered at each context size; the difference is not previously omitted text.",
              "These paired differences describe this fixed-seed cohort. Additional tuning labels and multiple comparisons "
              "limit claims about minimum data requirements or statistical significance."]
    matched = json.loads((study / "matched-coverage.json").read_text())
    parts += ["", "## Truncation and matched coverage", "",
              f"Qwen's 4096-token budget includes the prompt. Long bodies retain first/last tokens, omitting the middle. "
              f"The common fully covered subset contains {matched['documents']} documents, identical for every model.",
              table(["Model", "Matched F1", "Review-worthy support", "Dangerous misses"],
                    [[label(configs[n]), number(matched['models'][n]['overall'].get('macro_f1')),
                      matched['models'][n]['overall'].get('default_discard', {}).get('review_worthy_support', 0),
                      matched['models'][n]['overall'].get('default_discard', {}).get('dangerous_misses', 0)] for n in names]),
              "This subset can differ systematically from long documents; full-cohort results remain necessary. "
              "Encoder/Laya windows retain all source tokens. Qwen probabilities are conditional on A/B/C; "
              "full-vocabulary answer mass and retained-token fractions are saved per document.",
              "", "## Latency and cost", "", "![Warm throughput measured locally](throughput.png)"]
    timing = []
    for n in names:
        perf = json.loads((study / entries[n]["directory"] / "performance.json").read_text())
        timing.append([label(configs[n]), number(perf['latency_seconds']['p50']), number(perf['latency_seconds']['p95']),
                       number(perf['batch_throughput']['1']['documents_per_second']), number(perf['batch_throughput']['8']['documents_per_second']),
                       number(perf.get('peak_vram_bytes', 0) / 2**30, 2)])
    parts += [table(["Model", "P50 s", "P95 s", "Batch 1 docs/s", "Batch 8 docs/s", "Peak VRAM GiB"], timing),
              "Warm timings include preprocessing and all model windows, exclude loading, and use a fixed 128-document "
              "selection sample on a shared machine. Sklearn batches are native; neural/SDK/prompt document batches are serial, "
              "with window batching where supported. Local API charges are zero; hardware, electricity and collection are additional.",
              "The paid GPT-6.1 Sol reference was not executed. Using checked Flex prices without cache discounts, this test cohort "
              "would cost the following input/output scenarios. o200k tokenization is an approximation; the separate byte bound "
              "reserves prompt overhead and is deliberately conservative.",
              table(["Output tokens/doc", "Approximate USD", "Conservative USD"],
                    [[k, number(v['approximate_usd'], 2), number(v['conservative_usd'], 2)] for k, v in quote['scenarios_output_tokens_per_document'].items()]),
              f"Pricing checked {quote['pricing_checked']}: [official pricing]({quote['pricing_source']}). Reasoning tokens count as output. "
              "A future paid run needs an updated concrete quote and authorization; it has no measured accuracy or latency here.",
              "", "## Critical-error inspection", ""]
    resources = []
    for n in names:
        directory = study / entries[n]["directory"]
        perf = json.loads((directory / "performance.json").read_text())
        history = json.loads((directory / "history.json").read_text()) if (directory / "history.json").exists() else {}
        weights = cached_weights(configs[n])
        resources.append([label(configs[n]), number(history.get('seconds', configs[n].get('fit_seconds', 0)), 1),
                          number(perf.get('training_peak_vram_bytes', 0) / 2**30, 2),
                          number(weights / 2**30 if weights is not None else None, 2),
                          number(perf.get('adapter_bytes', perf.get('artifact_bytes', 0)) / 2**20, 2)])
    parts.insert(parts.index("## Critical-error inspection") - 1,
                 table(["Model", "Fit elapsed s", "Training VRAM GiB", "Original weights GiB", "Saved checkpoint MiB"], resources) +
                 "\n\nFit elapsed includes training/grid selection and recorded validation work; cached features can reduce elapsed time. "
                 "Original weights are cached pretrained input files, including unquantized Qwen weights, not packed deployment size. "
                 "Saved checkpoints may include optimizer/RNG state. Peak VRAM is torch-allocated memory, excluding driver/display overhead; "
                 "the latency table reports inference peak, with training peak listed separately.")
    features = [(name, configs[name]["feature_preparation"]) for name in names if configs[name].get("feature_preparation")]
    if features:
        parts.insert(parts.index("## Critical-error inspection") - 1,
                     table(["Model", "Feature documents", "Recorded compute s", "Feature peak GiB", "Cache reused"],
                           [[label(configs[name]), sum(v['documents'] for v in values.values()),
                             number(sum(v['recorded_compute_seconds'] for v in values.values()), 1),
                             number(max(v['peak_vram_bytes'] for v in values.values()) / 2**30, 2),
                             sum(v['reused'] for v in values.values())] for name, values in features]) +
                     "\n\nFrozen-feature counts include train, selection and calibration. Recorded compute sums each requested "
                     "document's original preprocessing/encoder time under the pinned recipe; it excludes model loading and "
                     "cache I/O. Cached rows retain this measurement when reused across sample sizes. Fit elapsed includes "
                     "only preparation performed during that fit. These timings describe the shared machine, not a new cold run.")
    by_hash = {body_hash(r): r for r in records}
    misses = sorted([r for r in predictions[primary] if r['reference'] == 'review_worthy' and r['prediction'] == 'routine'],
                    key=lambda r: r['probabilities'][0], reverse=True)[:8]
    for r in misses:
        source = by_hash[r['body_sha256']]
        excerpt = re.sub(r"\s+", " ", source['body'])[:450]
        parts += [f"Routine probability {r['probabilities'][0]:.3f}; {r['date']}; {r['family']}; "
                  f"[source]({source.get('url', '')}). Teacher label: review_worthy. Body hash: {r['body_sha256']}.", excerpt]
    if not misses:
        parts.append("No default review-worthy-to-routine misses were observed for the primary decision model on this finite cohort.")
    parts += ["Error examples are selected after testing for inspection, not model tuning. Labels may be wrong; "
              "analyst usefulness needs independent review. All miss identities and model disagreements remain inspectable in results.json.",
              "", "## Reproducibility and limits", "",
              "The serial runner stores failed/OOM configurations, any window-batch fallback, selected recipes, pinned model revisions, "
              "library versions, source snapshots, RNG/optimizer checkpoints, dataset/subset hashes and an immutable pre-test freeze. "
              "Reproduce with the locked dependencies and study.py --stage all; existing runs resume with changed-artifact guards. "
              "Raw-source rights/attribution travel with the corpus. Old API/container behavior remains separate.",
              "This is source-only inbox triage on constrained public cohorts, not expert materiality assessment or a trading strategy. "
              "One seed cannot quantify training instability. Bootstrap intervals quantify cohort sampling only. Teacher labels "
              "favor agreement with the labeling approach; the limited non-expert audit is reported separately. "
              "Missing source/date coverage, approximate duplicate retrieval, pretraining exposure and long-document clipping restrict generalization. "
              "No paid SOTA comparison or RL/RLCD result is claimed. RL/RLCD remains a possible follow-on after the core study."]
    if audit["resolved"]:
        parts += ["", "## Independent non-expert audit", "",
                  f"Resolved {audit['resolved']} documents; unresolved {audit['unresolved']}; teacher agreement {number(audit['teacher_agreement'])}.",
                  table(["Model", "Human macro F1", "Dangerous misses", "Review-worthy support"],
                        [[label(configs[n]), number(audit['models'][n]['overall']['macro_f1']),
                          audit['models'][n]['overall']['default_discard']['dangerous_misses'],
                          audit['models'][n]['overall']['default_discard']['review_worthy_support']] for n in names])]
    content = "\n\n".join(parts) + "\n"
    (output / "report.md").write_text(content, encoding="utf-8")
    pdf(content, output / "report.pdf")
    save({"freeze_sha256": fingerprint(study / "freeze.json"), "results_sha256": fingerprint(study / "results.json"),
          "report_source_sha256": fingerprint(HERE / "report.py"),
          "versions": {n: version(n) for n in ('matplotlib', 'reportlab', 'pypdf')},
          "outputs": {name: fingerprint(output / name) for name in
                      ('report.md', 'report.pdf', 'human-audit.json', 'reference-quote.json',
                       'learning-curves.png', 'context.png', 'throughput.png', 'confusion.png')}}, output / "manifest.json")
    print(f"Report written to {output}; render and inspect PDF before publication", flush=True)
