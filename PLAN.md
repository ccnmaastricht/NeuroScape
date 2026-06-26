# Dataset Update Plan

## PR #1 (Maldinni)

Review written to `pr1_review.md`. Three fixes requested before merge:

1. Revert `config/analysis/semantic.toml` to original values (`model_name`, `max_abstract_number`, `temperature`, `retries`) — the Colab notebooks don't read this file, so the changes only break the production scripts.
2. Revert `config/preprocessing/classifier.toml` `num_classes` back to 26.
3. Fix `drop_class` guard in `src/utils/classifier.py`: replace the compound condition with `if X.shape[0] == 0:`.

The PR's Colab ingestion notebook also contains a useful `citation_rate` computation (see Phase 0 below).

---

## Incremental Update Strategy

Add new data annually via KNN assignment to existing clusters. Full re-cluster only every ~5 years (retrain embedding model → rebuild graph → community detection → re-run all semantic analysis).

---

## Phase 0 — Fix production code

1. **`scripts/semantic_analysis/assess_dimensions.py:24`** — remove hardcoded `BASEPATH = '/media/mario/HDD/Data/NeuroScape_copy'`.

2. **`scripts/ingestion/build_adjacencies.py` + `src/utils/adjacency.py`** — extract the citation_rate computation from the PR's Colab ingestion notebook into the production pipeline, replacing the fragile CrossRef fetch in `article_metadata.py`:
   ```python
   article.citation_rate = len(article.in_links) / max(1, end_year - article.year)
   ```
   Use `--end_year` argument as the reference year, not a hardcoded value.

---

## Phase 1 — Scrape 2024–2025

```bash
python scripts/ingestion/scraping.py --discipline Neuroscience --start_year 2024 --end_year 2025
```

Neuroscience only — classifier is not being retrained so other disciplines are not needed.

---

## Phase 2 — Merge, clean, embed (new articles only)

```bash
python scripts/ingestion/merge_and_clean.py
python scripts/ingestion/initial_embedding.py       # Voyage AI, new articles only
python scripts/preprocessing/filter_disciplines.py  # existing classifier applies fine
python scripts/ingestion/build_adjacencies.py       # now also computes citation_rate
```

Note: `citation_rate` for new articles will be low initially (few citing papers in the dataset yet). `citation_rate` of existing 1999–2023 articles will not reflect citations from 2024–2025 papers. Acceptable for cluster assignment purposes.

---

## Phase 3 — Domain embed new articles

```bash
python scripts/domain_embedding/embed_abstracts.py  # existing trained model, new articles only
```

No retraining of the `SparseEmbeddingNetwork`.

---

## Phase 4 — Assign to existing clusters (new script needed)

New script: `scripts/clustering/assign_to_clusters.py`

- Load existing domain embeddings + cluster labels for 1999–2023 articles
- Compute cluster centroids (mean of L2-normalised embeddings per cluster)
- For each new article, assign to nearest centroid by cosine similarity
- Append assignments to the main CSV

Nearest centroid is preferred over full KNN — 461K articles makes KNN over the full corpus expensive, and centroids capture cluster geometry well since embeddings are L2-normalised.
