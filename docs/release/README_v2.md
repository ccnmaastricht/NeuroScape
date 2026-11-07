# NeuroScape 2.0

> **Draft (2026-11-07).** Values marked `TBD` are filled in after the January 2027 citation refresh and the final
> assembly (`assembly_report.json`). Items marked **Decision** are still open.

NeuroScape is a dataset of neuroscience journal articles from PubMed with abstracts, domain-specific embeddings,
a citation graph and a map of 175 research clusters. Version 2.0 extends version 1.0.1
(1999–2023, 461,316 articles) to **1980–2025** and **TBD articles**.

- Concept DOI: [10.5281/zenodo.14865161](https://doi.org/10.5281/zenodo.14865161) (all versions)
- Code: <https://github.com/MSenden/NeuroScape>. The tag `aperture-neuro` reproduces the paper (v1.0.1); NeuroScape 2.0 was
  built on branch `dataset-v2` (**TBD**: release tag).
- Cite: Senden, M. (2027). NeuroScape (2.0) [Data set]. Zenodo. **TBD**: version DOI.

## What is new in 2.0

| | v1.0.1 | v2.0 |
|---|---|---|
| Years | 1999–2023 | 1980–2025 |
| Articles | 461,316 | TBD (provisional: 689,022) |
| Citation links (article graph) | 7,702,484 | TBD (provisional: 11,533,443) |
| Citation counts | fetched around January 2025 | refetched for all articles in January 2027 |
| Clusters | 175 (Leiden) | the same 175; new articles are assigned to them |

Every article has an `Added In` value:

| `Added In` | Years | Meaning | Articles (provisional) |
|---|---|---|---|
| `v1.0` | 1999–2023 | in v1.0.1; all v1 columns unchanged except citation fields | 461,316 |
| `v2.0` | 2024–2025 | new years | 60,993 |
| `v2.0-late` | 1999–2023 | indexed in PubMed after the v1 scrape | 18,184 |
| `v2.0-early` | 1980–1998 | before Scimago coverage; proxy journal lists (see below) | 148,529 |

## How articles were selected

1. **Journals.** For 1999–2025, Neuroscience journals in Scimago Q1 or Q2 and Multidisciplinary journals in Scimago Q1,
   taken from each year's Scimago list (the v1 rule).
2. **1980–1998 (proxy).** Scimago starts in 1999. For earlier years the journal set is the journals that meet the
   rule in at least 3 of 1999–2003. Journals that were renamed are followed back to their earlier titles in the NLM
   Catalog (e.g. *Electroencephalography and Clinical Neurophysiology* → *Clinical Neurophysiology*). Each year includes
   only the journals published in that year.
3. **Articles.** All PubMed records of these journals in the year (at most 5,000 per journal and year), restricted to
   research articles and reviews with an abstract of 50–500 words.
4. **Discipline filter.** A classifier trained on Voyage `voyage-lite-02-instruct` embeddings keeps the articles
   predicted to be neuroscience (confidence ≥ 0.8). This mainly removes non-neuroscience articles from
   Multidisciplinary and multi-area journals.
5. **Cut-off.** Articles with a 2026 issue date (published online in 2025) are not included. They will enter with
   the next update.

## Limitations of the 1980–1998 extension

- **Journal coverage is a proxy.** Journals that ceased, or fell below Q2, before 1999 are missing, so early
  volumes are biased towards journals that were still prominent in 1999–2003.
- **Abstract coverage in PubMed is incomplete for 1980–1984.** Articles without an abstract are excluded throughout
  NeuroScape, so these years are undercounted, and the shortfall varies by journal. From 1985 coverage is essentially
  complete.
- **Clusters were defined on 1999–2023.** Early articles are assigned to the nearest existing cluster. Topics that
  faded before 1999 have no cluster of their own, so early articles fit less well (median `Assignment Share`: 0.70 in
  1980–84 vs. 0.82 in 1995–98 and 0.84 in 2024–25).
- **Growth over 1980–2025 is not comparable to v1.** Neuroscience grew about 9.6% per year in 1980–1998 and about 4.4%
  per year in 1999–2025, so linear growth measures depend strongly on the year window. **Decision:** which window the
  growth classification in the documentation uses (1999–2025, comparable to v1, or 1980–2025).

## Files

| File | Covers | Notes |
|---|---|---|
| `CSV/neuroscience_articles_1980-2025.csv` | all articles | one row per article; v1 rows first in v1 order, then added articles by Year, Journal, Type |
| `CSV/neuroscience_clusters_1980-2025.csv` | 175 clusters | statistics recomputed over all articles; text columns from v1 |
| `HDF5/DomainEmbeddings/shard_*.h5` | all articles | same order as the article CSV, 200 per shard |
| `Graphs/article_citation.graphml` | all articles | directed citation graph (citing → cited), within the dataset |
| `Graphs/cluster_citation_density.graphml` | 175 clusters | citation density between clusters |
| `Graphs/article_similarity.graphml` | **v1 articles only** | carried over from v1.0.1 |
| `CSV/neuroscience_dimensions_1999-2023.csv`, `CSV/global_trends.csv`, `CSV/just_trends.csv` | v1 | LLM-derived; carried over from v1.0.1 |
| `Models/discipline_classification_model.pth`, `Models/domain_embedding_model.pth` | – | unchanged from v1.0.1 |
| `assembly_report.json` | – | counts per source and year, link totals, reference date, notes |

**Decision:** v1.0.1 also contained the raw Voyage embeddings (`HDF5/VoyageAIEmbeddings/Large_02_Instruct`,
`Lite_02_Instruct`; 6.4 GB). They exist for all v2 articles but are not yet part of the v2 assembly.

### Article table (`neuroscience_articles_1980-2025.csv`)

| Column | Meaning |
|---|---|
| `Pmid`, `Doi` | PubMed ID and DOI |
| `Type` | `Research` or `Review` |
| `Title`, `Abstract`, `Journal` | from PubMed |
| `Year`, `Month` | publication date from PubMed |
| `Disciplines` | Scimago subject areas of the journal |
| `Cluster ID` | cluster (0–174). v1 articles: Leiden community; added articles: assigned (see below) |
| `Citations` | CrossRef `is-referenced-by-count` |
| `Citations Fetched` | date the count was fetched |
| `Age` | years from publication (year and month) to the reference date **2027-01-01** |
| `Citation Rate` | `Citations / Age` |
| `Added In` | see above |
| `Assignment Share`, `Assignment Margin` | added articles only (empty for v1): similarity-weighted vote share of the assigned cluster among the 50 nearest v1 articles, and its lead over the runner-up |

Ages and citation rates use a fixed reference date rather than the fetch date, so they don't change with the date
on which the data were processed.

### Cluster table (`neuroscience_clusters_1980-2025.csv`)

Recomputed over all v2 articles: `Size`, `Year First Article`, `MCR Research`, `MCR Review` (median citation rate of
research articles and reviews), `Reference Krackhardt` and `Citation Krackhardt` (Krackhardt E/I index of a cluster's
references and citations; positive = mostly to/from other clusters), `Most Cited Cluster` and `Most Citing Cluster`.

Unchanged from v1 (LLM-generated from v1 articles): `Title`, `Keywords`, `Description`, `Focus`, `Most Similar
Cluster`, `Similarity`, `Distinguishing Features`, `Open Questions`, `Dimensions`, `Trends`.

### HDF5 shards

Same layout as v1.0.1. Per shard, datasets `pmid`, `doi`, `title`, `abstract`, `journal`, `type`, `year`, `age`,
`citation_count`, `citation_rate`, `in_links`, `out_links` and a group `embeddings` with one 64-d domain embedding per
article. A missing citation count is stored as `-1` (NaN in the CSV). `in_links`/`out_links` are PMIDs within the dataset.

## Methods for added articles

- **Embeddings.** Same models as v1: `voyage-lite-02-instruct` for the discipline filter, `voyage-large-2-instruct` as
  input to the (unchanged) domain-embedding network (1024 → 64 dimensions). Neither model was retrained.
- **Cluster assignment.** Similarity-weighted vote among the 50 nearest v1 articles in the domain-embedding space
  (cosine similarity). The neighbours are v1 articles only, so the v1 clustering is the fixed reference and added
  articles don't influence each other. Validation: assigning the v1 2023 articles using only 1999–2022 articles
  reproduced their Leiden labels for 95.9% of articles; for articles with `Assignment Share` ≥ 0.7, agreement was
  100%. A centroid method reached 76.5% and was not used.
- **Citation links.** Citing articles from PubMed (`pubmed_pubmed_citedin`) and references from CrossRef, both
  restricted to articles in the dataset, as in v1. Reference DOIs are matched case-insensitively for added articles,
  so their links are slightly more complete than v1's, which were not recomputed. Links are symmetric
  (an out-link of A to B is an in-link of B).
- **Citation counts.** Refetched from the CrossRef REST API for all articles in January 2027 (`Citations Fetched`).

## Changelog

### 2.0 (TBD, January 2027)
- Extended to 1980–2025 (`v2.0-early`, `v2.0`, `v2.0-late`).
- Citation counts refetched for all articles, with a fixed reference date (2027-01-01) for ages and citation rates.
- Article citation graph rebuilt over all articles. The local v1 copy of `article_citation.graphml` was incomplete;
  this file was not part of the v1.0.1 Zenodo record.
- Cluster statistics recomputed over all articles; cluster definitions and texts unchanged.
- New columns: `Added In`, `Assignment Share`, `Assignment Margin`, `Citations Fetched`.

### 1.0.1 (2025)
- Initial public release, 1999–2023.
