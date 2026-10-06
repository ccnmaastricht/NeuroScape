# Spec: NeuroScape 2.0 — dataset update with 2024–2025

Status: **agreed** (spec interview 2026-10-06) · Supersedes the incremental-update part of `PLAN.md`

## 1. Goal

Publish **NeuroScape 2.0** on Zenodo (new version under the v1 concept DOI) covering
1999–2025. Do all scraping and processing now (Oct 2026). In January 2027, refresh
citations and release.

## 2. Constraint: the Aperture Neuro paper stays reproducible

- **Code.** Tag current `main` (`3612330`) as `aperture-neuro`. That tag plus Zenodo 1.0.1
  is the reproduction recipe, and the README says so.
- **Git.** All work happens on branch `dataset-v2`. Shared scripts in `scripts/` and `src/`
  only gain options whose defaults keep v1 behaviour. New steps get new scripts. Paper
  notebooks `results_00`–`03` are not edited; v2 checks go in new notebooks.
- **Data.** All work happens in a new root, `/media/mario/HDD/Data/NeuroScape_v2`. The
  pipeline is never run with `BASEPATH` pointing at `NeuroScape/` or `NeuroScape_original/`.
- **Protection** (all three):
  1. SHA-256 manifest of `NeuroScape/Public` and `NeuroScape_original` before starting,
     checked again before merging.
  2. `chmod -R a-w` on both.
  3. At the end, run `results_00`–`03` at the tag on v1 data and compare with the saved SVGs
     in `Internal/Manuscript/Figures`.
- Merge `dataset-v2` into `main` only after all three checks pass.

## 3. Current state (inventory 2026-10-06)

- **v1** (461,316 articles, 1999–2023) is intact only in `NeuroScape/Public` and
  `NeuroScape_original` (`NeuroScape_v101.zip`). `NeuroScape/Internal` was overwritten by the
  Oct 2025 partial run and is not v1.
- **Two Voyage embedding spaces.** Both are 1024-d and incompatible with each other
  (cosine ≈ 0 for the same article). Both models are still served (checked 2026-10-06).
  - `voyage-lite-02-instruct` → input to the discipline classifier
    (`discipline_classification_model_finetuned.pth`, class index 13, cutoff 0.8)
  - `voyage-large-2-instruct` → input to the domain-embedding model
    (`domain_embedding_model_best.pth`, 1024→64)
- **Oct 2025 delta** in `NeuroScape/Internal/Intermediate/HDF5/*`: 44,905 articles. All are
  lite-embedded, filtered, large-embedded and domain-embedded; none have links yet.
  - 26,721 are from 2024.
  - ≈18,200 are from 1999–2023 but missing from v1 ("late additions").
- **Scimago** lists exist for Neuroscience and Multidisciplinary up to 2024. The 2025 lists
  still need downloading.

## 4. Pipeline (all in `NeuroScape_v2`)

**Phase 0 — Setup**
1. Run the protection steps (§2) and tag `aperture-neuro`.
2. Seed the v2 root:
   - v1 public CSV/HDF5/graphs
   - both models, `Internal/Reference` (incl. `journal_lut.csv`)
   - the Oct 2025 delta: CSVs, both Voyage HDF5 sets, domain HDF5
   - checkpoints (`scraped_articles.json`, `embedded_articles.json`)
3. Download the Scimago 2025 lists (Neuroscience, Multidisciplinary).
4. Record both model names in config:
   - `initial_embedding.toml`: `model` (lite) and `domain_input_model` (large)
   - `update_embedding.py` reads the large model from config, not from its CLI default.
5. Pin the environment (`requirements-v2.lock`) and record package versions.

**Phase 1 — Scrape**
- Neuroscience and Multidisciplinary journals, same Scimago quartile rule as v1.
- 2024: top-up only. The checkpoint skips PMIDs already processed, so this only adds
  articles indexed since Sep 2025. 2025: full scrape.
- `scraping.py` must honour `--start_year/--end_year` (default 1999–2023 = v1 behaviour).

**Phase 2 — Clean, embed, filter (new articles only)**
1. Clean: `year_cutoff = 2025`, same word limits.
2. Dedupe against v1 and the delta by PMID and DOI.
3. Embed with lite.
4. Filter with the classifier. New shards are numbered after the existing ones; nothing is
   rewritten.
5. Embed the survivors with large.
6. Late additions (year ≤ 2023, not in v1) are kept and flagged (§5).

**Phase 3 — Domain embedding.** Run the existing `SparseEmbeddingNetwork` on new articles.
No retraining.

**Phase 4 — Citation links**
- Fetch in/out links for new articles only.
- Add back-links to v1 articles' `in_links` by symmetry. No refetch for v1 articles.

**Phase 5 — Cluster assignment** (new script `scripts/clustering/assign_to_clusters.py`)
- Method: nearest centroid in cosine space on L2-normalised domain embeddings. Centroids
  come from v1 members.
- Stored per article: `Cluster ID`, similarity to the assigned centroid, and the margin to the
  second-best centroid.
- **Validation first:** reassign the v1 2023 articles and compare with their Leiden labels.
  Accept if agreement is **≥ 80%**; otherwise switch to a kNN majority vote and re-validate.

**Phase 6 — Assemble 2.0 (pre-release)**
- `neuroscience_articles_1999-2025.csv`, HDF5 (domain embeddings + links),
  `article_citation.graphml`, `cluster_citation_density.graphml`.
- `neuroscience_clusters_1999-2025.csv`: v1 definitions unchanged; sizes, growth and citation
  stats recomputed over **all** v2 articles, late additions included.
- `article_similarity.graphml` and the dimensions/trends files are carried over from v1.

**Phase 7 — January citation refresh (build and dry-run now)**
- New script that refetches CrossRef counts for **all** 1999–2025 articles.
- Stores `Citations` and `Citations Fetched` (date).
- `Citation Rate = Citations / age`, where age runs from publication to a **fixed reference
  date** (2027-01-01) stored in the dataset metadata, not the run date. The v1 definition is
  otherwise unchanged.
- Refetch in/out links for 2024–2025 articles once more.
- Must be resumable (checkpointed). Dry run on ~1,000 articles now; full run in January.
- Then: rerun Phase 6 stats and graphs, write the README/changelog for Zenodo, and release.

## 5. Data model additions (2.0)

| Column | Meaning |
|---|---|
| `Added In` | `1.0` (in v1), `2.0` (2024–2025) or `2.0-late` (1999–2023 added in 2.0) |
| `Assignment Similarity`, `Assignment Margin` | Only for articles assigned in 2.0. v1 articles keep their Leiden labels (NaN) |
| `Citations Fetched` | Date the CrossRef count was fetched |

The citation-rate reference date goes in the release README.

## 6. Acceptance criteria

- [ ] v1 manifests unchanged, read-only, paper figures reproduce at `aperture-neuro`
- [ ] No PMID/DOI duplicates; v1 rows identical to v1 except `in_links` and citation fields
- [ ] 2024/2025 article counts per journal are plausible against 2021–2023 (sanity notebook)
- [ ] Centroid assignment ≥ 80% agreement on v1 2023 articles
- [ ] Citation-refresh dry run succeeds and resumes after interruption
- [ ] All new and changed scripts reproduce v1 behaviour with default arguments

## 7. Out of scope

- Retraining the classifier or the domain model, and re-clustering. A full re-cluster comes
  later and may use the late additions.
- Rerunning the LLM dimensions/trends.

## 8. Contingency

If a Voyage model is retired mid-run:
- **lite gone:** replace the discipline filter with a classifier trained on large embeddings
  and keep the domain model.
- **large gone:** blocker; stop and discuss.
