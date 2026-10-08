# Public-source audit

Checked 2026-10-08. Counts below come from the collectors' saved indexes and extraction pilots, not estimates
of the number of usable labeled examples. The final corpus census is produced after collection and duplicate
auditing. Original responses, retrieval times and hashes remain in the local response cache.

| Family | Access and implementation | Reuse conditions | Observed coverage |
| --- | --- | --- | --- |
| Primary 8-K | [EDGAR](https://www.sec.gov/edgar/search/), [edgartools](https://github.com/dgunning/edgartools) 5.59.1, MIT SDK | [Public filing content is free to access and reuse](https://www.sec.gov/files/about/webmaster-faq.htm); exclude images | 117,209 indexed in Jan 2025–Sep 2026; 50-document pilot |
| Corporate releases | EX-99 release/earnings/announcement exhibits attached to 8-Ks; same SDK | Same SEC policy; retain the filing accession for event grouping | 49 accepted after one MD&A exhibit was excluded from a 50-document pilot; attachment availability limits yield |
| 6-K | EDGAR primary documents; same SDK | Same SEC policy | 199,378 indexed in Jan 2019–Sep 2026; 50-document pilot |
| Federal Reserve | Annual press-release and speech HTML archives; requests/BeautifulSoup | [Public domain unless otherwise indicated](https://www.federalreserve.gov/disclaimer.htm); attribution, exclude third-party media | 1,831 indexed in Jan 2019–Sep 2026; 50-document pilot |
| ECB | Versioned JSON archive used by its website, followed by English institutional HTML | [Attribution, accuracy and stated modifications](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html); author-named documents excluded | 722 eligible archive entries in Jan 2019–Sep 2026; 50-document pilot |
| General news | [Wikinews category API](https://www.mediawiki.org/wiki/API:Categorymembers), VOA archive, GDELT URL discovery | [Wikinews CC BY 4.0 / earlier CC BY 2.5](https://en.wikinews.org/wiki/Wikinews:Copyright); [VOA-original text only](https://www.voanews.com/p/5338.html); agency and mixed-rights text excluded | 2,356 discovery candidates, mostly historical Wikinews entries; 50-document pilot, all Wikinews; full coverage filtering pending |

Fed and ECB collection uses their article archives rather than economic-series SDKs. GDELT has a
[MIT Python client](https://github.com/alex9smith/gdelt-doc-api), but this implementation uses its small REST
endpoint directly. GDELT's article search is a recent, capped discovery feed; its URLs confer no publisher
license. MediaWiki is queried directly to avoid an extra SDK dependency.

These sources are free to fetch, with different redistribution obligations. The collectors retain attribution
and modification notices per row. A release and its covering filing share an event ID. Primary 6-Ks may be
short cover sheets whose substance resides in an uncollected exhibit: this is visible source content and can
legitimately be unclear. It must be measured rather than silently filled with external context.

The news sample is a constrained public corpus, not a representative commercial financial-news feed.
Publication timestamps must come from the article itself; discovery times are never used in their place.
Historical archive pages can change after publication, so retrieval snapshots do not prove that the extracted
text was available verbatim on its stated publication date. Pretrained-model exposure remains possible.

## Pilot and implementation checks

- Twelve Luna Flex labels, two per family, completed for $0.00105404. Four capacity rejections had zero charge.
  This verifies dispatch, structured labels and usage accounting; it does not validate the annotation rubric.
- Shared-budget reservations, interruption recovery, metadata round trips and human-label precedence passed.
- Synthetic event/duplicate groups crossing time boundaries were quarantined; old experiment bodies were
  excluded from fresh validation/test. Frozen splits rejected changed outputs.
- Blind review forms produced 72 unique evaluation documents and 12 overlaps; conflicting reviewers remained
  unresolved. Conflicting exports from one reviewer are rejected.
- The original archive's split files and manifest reproduced byte for byte. Ruff, compilation, CLI parsing and
  the locked dependency resolution passed. Neural and full-corpus experiments have not yet run.
- Docker's local daemon is unavailable; container build and inference are left to the existing CI workflow.

Collection and the independent human audit remain unfinished. Dataset-size targets are ceilings, and source
shortfalls will be recorded explicitly before model comparisons.

The first complete news pass retained 83 Wikinews articles dated Jan 2019–Apr 2026, with no eligible VOA
articles. A broader pass through two levels of financial/business subcategories is running before the news
source is declared exhausted. The completed Fed corpus has 1,826 documents; ECB has 722. Data-PR CI lint,
container build and container inference passed despite the unavailable local Docker daemon.
