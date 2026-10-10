# Public-source audit

Access and reuse policies checked 2026-10-09; collection census updated 2026-10-10.
All seven collection passes have completed: 89,006 new documents plus the 10,000-document legacy archive,
for 99,006 raw candidates. These are acquisition counts, not labeled examples or independent evaluation
support. Original responses, retrieval times and hashes remain in the local response cache.

| Family | Access and implementation | Reuse conditions | Observed coverage |
| --- | --- | --- | --- |
| Primary 8-K | [EDGAR](https://www.sec.gov/edgar/search/), [edgartools](https://github.com/dgunning/edgartools) 5.59.1, MIT SDK | [Public filing content is free to access and reuse](https://www.sec.gov/files/about/webmaster-faq.htm); exclude images | Target reached: 30,000 accepted from 540,703 indexed in Jan 2019–Sep 2026; reproducible month-balanced sample |
| Corporate releases | EX-99 release/earnings/announcement exhibits attached to 8-Ks; same SDK | Same SEC policy; retain the filing accession for event grouping | Target reached: 30,000 accepted from 244,685 indexed in Jan 2019–Sep 2026; query-defined exhibit sample, not a release census |
| 6-K | EDGAR primary documents; same SDK | Same SEC policy | Target reached: 25,000 accepted from 199,378 indexed in Jan 2019–Sep 2026; reproducible month-balanced sample |
| Federal Reserve | Annual press-release and speech HTML archives; requests/BeautifulSoup | [Public domain unless otherwise indicated](https://www.federalreserve.gov/disclaimer.htm); attribution, exclude third-party media | Complete pass: 1,826 accepted from 1,831 indexed in Jan 2019–Sep 2026; 55 in Q3 2026 |
| ECB | Versioned JSON archive used by its website, followed by English institutional HTML | [Attribution, accuracy and stated modifications](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html); author-named documents excluded | Complete pass: 722 accepted in Jan 2019–Sep 2026; 27 in Q3 2026 |
| Public news | [Wikinews category API](https://www.mediawiki.org/wiki/API:Categorymembers), VOA archive, GDELT URL discovery | [Wikinews CC BY 4.0 / earlier CC BY 2.5](https://en.wikinews.org/wiki/Wikinews:Copyright); [VOA-original text only](https://www.voanews.com/p/5338.html); agency and mixed-rights text excluded | Broader pass: 201 accepted from 4,535 candidates; all Wikinews, Jan 2019–Apr 2026, none in Q3 2026 |
| Official financial news | HM Treasury articles: GOV.UK Search API discovery and supported Content API extraction; requests/BeautifulSoup | [Open Government Licence v3.0, except where otherwise stated](https://www.gov.uk/help/reuse-govuk-content); retain attribution, omit assets and special-rights notices | Complete pass: 1,257 accepted from 1,329 candidates in Jan 2019–Sep 2026, including 35 in Q3 2026; 72 later-revised articles excluded |

Fed and ECB collection uses their article archives rather than economic-series SDKs. GDELT has a
[MIT Python client](https://github.com/alex9smith/gdelt-doc-api), but this implementation uses its small REST
endpoint directly. GDELT's article search is a recent, capped discovery feed; its URLs confer no publisher
license. MediaWiki is queried directly to avoid an extra SDK dependency.
GOV.UK explicitly supports its Content API but calls the Search API unsupported; saved discovery indexes
make the cohort inspectable if that interface changes. Both are accessed directly without an additional SDK.
HM Treasury articles are official announcements, not a substitute for independent financial journalism.
Their Content API's original publication timestamp determines the date; articles publicly revised after
that publication day are excluded because the original body is unavailable. Images and attachments are omitted.

These sources are free to fetch, with different redistribution obligations. The collectors retain attribution
and modification notices per row. A release and its covering filing share an event ID. Primary 6-Ks may be
short cover sheets whose substance resides in an uncollected exhibit: this is visible source content and can
legitimately be unclear. It must be measured rather than silently filled with external context.

SEC document dates are filing dates from the SEC index, including dates of filings that carry release
exhibits. They establish EDGAR disclosure dates, not the earliest publication elsewhere: the Armlogi
pilot release is dated January 14 in its text and January 15 in its covering filing. Other source dates
come from original article dates or publication metadata; discovery times are never substituted.
Temporal splits use these recorded document dates, with day precision and the stated availability limits.
The news sample is a constrained public corpus, not a representative commercial financial-news feed.
Historical archive pages can change after publication, so retrieval snapshots do not prove that the extracted
text was available verbatim on its stated publication date. Pretrained-model exposure remains possible.

## Pilot and implementation checks

- Twelve Luna Flex pilot labels, two per family, cost $0.00123459 after correcting the ledger for cache writes.
  Four capacity rejections had zero charge. This verifies dispatch, structured labels and usage accounting;
  it does not validate the annotation rubric. Main labeling uses the shared approved $5 maximum.
- Shared-budget reservations, interruption recovery, metadata round trips and human-label precedence passed.
- Offline server-error checks verified one separately reserved retry, its restart bound, retained uncertain
  charges, zero-charge capacity rejections and refusal to exceed the shared ceiling; no API calls were made.
- Synthetic event/duplicate groups crossing time boundaries were quarantined; old experiment bodies were
  excluded from fresh validation/test. Frozen splits rejected changed outputs.
- Offline form/import checks verified blinded assignments, rejected conflicting exports from one reviewer
  and left synthetic reviewer disagreements unresolved. The fresh pack now contains 72 unique test documents,
  12 overlaps and 42 assignments per reviewer. Fresh human responses remain pending.
- The original archive's split files and manifest reproduced byte for byte. Ruff, compilation, CLI parsing and
  the locked dependency resolution passed. Main-corpus model-quality experiments remain pending.
- Docker's local daemon is unavailable; container build and inference are left to the existing CI workflow.

Collection is complete. Versioned Luna evaluation labels are complete; training labeling and the benchmark
freeze remain in progress at this census update. The independent non-expert human audit is pending.
Dataset-size targets are ceilings; final labeled support and source shortfalls precede model comparisons.

The broader news pass through two levels of financial/business subcategories is complete. Its 201 articles
do not establish recent-news generalization: Wikinews contributes no Q3 2026 test rows, and no eligible VOA
articles were found. HM Treasury extends the news family with official announcements; provider counts and
metrics remain separate. The completed Fed corpus has 1,826 documents; ECB has 722. Data-PR CI lint,
container build and container inference passed despite the unavailable local Docker daemon.
An actual 12-document blinded rubric pilot is available locally under `data/research/rubric-pilot/`.
Its teacher labels are hidden; it is a rubric check, not the later independent test audit.
The returned non-expert review contains 12 answers and no explanatory notes. Eight agree with Luna;
four differ on a thin 6-K cover sheet, administrative amendments, a director appointment and an ECB
statistical release. The reviewer explicitly cautioned that the exercise was difficult. Answers and
their source mapping are preserved locally as rubric feedback; they do not override dataset labels
or provide held-out accuracy evidence. These cases need interpretation before changing the rubric.
The reviewer subsequently identified themselves as a non-analyst and preferred Luna as the labeller.
The existing rubric and Luna reference labels remain in place; this pilot informs task clarity rather than expert accuracy.

Source-date, language, required provenance and body/raw-response hash checks passed on all 89,006 new
documents; their bodies and dates reproduce offline from retained responses. The full 99,006-record raw
archive also passed the workflow hash audit. Local receipts are `data/research/checks/final-corpus-source-audit.json`,
`completed-source-reextraction.json` and `release-prefix-latest.json`. Three Wikinews pilot dates/licence fields
were repaired from preserved responses without changing bodies, identities or training-period membership.
News snapshots also use the retained pilot cache and earlier raw files; these must travel with the corpus
for reproduction. No main benchmark has yet been frozen.
Valid source text can contain Unicode line separators, so JSONL readers split only at the record newline.
Preparation records raw input counts and publication ranges, then exclusions by family; acquisition
counts must not be presented as labeled examples or independent evaluation support.

The legacy archive used edgartools primary-document text, while the new corpus uses HTML text extraction.
An offline check of 1,485 validation/test-date candidates found 218 exact old-body matches, including
64 below .90 five-word-shingle Jaccard similarity. Reconstruction uses the retained primary HTML and
the pinned SDK's text formatter; it does not infer shared identity from issuer names. Preparation now
adds those exact-format links to the event/duplicate groups before split selection, including related
release exhibits. It records the SDK version, attempted/matched counts and body/raw hashes without
changing canonical source bodies. This closes verified cross-format overlaps; other duplicate retrieval
remains approximate. Missing or changed retained raw responses stop reconstruction before labeling.

Primary sampling alone cannot cover every release's parent filing. Three retained earnings/management
release exhibits dated May/August 2026 demonstrated old filing events whose primaries were absent from
the collected 8-K prefix. The SDK transport cache supports recovery of all 10,000 old body identities by
exact hash, including six exceptional formats. The recovery snapshot preserves complete submissions and
required index/primary responses, formatter methods, original cache metadata and checksums. A complete
primary preceding one truncated trailing attachment is usable only when it reproduces the old body exactly;
the damaged source bytes remain preserved. Preparation verifies the snapshot independently and joins
registered filing events before temporal selection. The registry travels with the original archive and corpus.

Description-only release discovery overrepresented certain filing templates: SEC full-text results include
major-company release exhibits whose descriptions contain only EX-99.1. Release discovery therefore now uses
full-text press/news-release and distribution-wire markers, filtered to EX-99 exhibits. Queries are split by
month and further subdivided for capped searches or repeated server errors. Unrecoverable single-day
failures still reject discovery. This is a reproducible query
cohort, not a census of every corporate release. Twelve exhibits from interrupted monthly searches were
absent from completed smaller-range retries; their original responses remain preserved. Discovery pages
span all approved dates but do not establish completeness. The earlier pilot remains an extraction check.
The initial grouped OR query returned zero: EFTS interpreted its parentheses as required literal terms.
The corrected ungrouped query returned 4,896 January 2019 hits, including EX-99 exhibits. The empty index
and original responses are retained as failed discovery evidence; the corrected query is bound to collection
settings. Search timeouts, shard failures and incomplete pages reject an index rather than hiding shortfalls.

Primary SEC document discovery now reads the canonical filing index before falling back to the SDK's
full-submission lookup. On 12 collected 8-Ks and 12 collected 6-Ks, both routes selected the same primary
document URL; the index response and its hash are retained. This reduces unnecessary exhibit downloads
without changing which document is collected. An interrupted final JSONL append is archived before
recovery, while malformed interior records still reject the file. Completed rows reconcile a lagging
collection checkpoint so a resumed job keeps their accepted status.
