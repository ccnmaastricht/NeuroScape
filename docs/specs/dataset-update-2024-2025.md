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

## 9. Progress log

### Phase 0 — done 2026-10-06

- Tag `aperture-neuro` → `3612330`; work on branch `dataset-v2`.
- Manifests: `NeuroScape_v2/Manifests/v1_NeuroScape_Public.sha256` (6,944 files) and
  `v1_NeuroScape_original.sha256` (13,514 files). `NeuroScape/Public`, `NeuroScape_original`,
  the manifests and `Base/V1` are read-only (`chmod -R a-w`).
- **Finding:** `NeuroScape/Public/Data/CSV/neuroscience_dimensions_1999-2023.csv` was overwritten
  in Oct 2025 (NatureTalk re-run). The v1 file is preserved as `... (copy).csv`, byte-identical
  to `NeuroScape_original`. All other `Public/Data` files match `NeuroScape_original`.
  **Fixed 2026-10-06:** the v1 file is restored under its original name and the NatureTalk version is
  renamed to `neuroscience_dimensions_1999-2023_naturetalk.csv`. The `NeuroScape/Public` manifest was
  regenerated. All v1 files now match `NeuroScape_original`; the only extras are
  `dimensions_all.csv` and the `_naturetalk` file.
- Voyage models were verified by re-embedding stored abstracts (cosine 1.0): delta
  `VoyageAIEmbeddingsOriginal` = `voyage-lite-02-instruct`, delta `VoyageAIEmbeddings` =
  `voyage-large-2-instruct`. The old config value `voyage-lite-2-instruct` is not a valid model name.
  Both names are now in `config/ingestion/initial_embedding.toml`.
- Environment: conda env `neuroscape_v2` (Python 3.12, torch 2.7.1+cu126), pinned in
  `requirements-v2.lock`. `requirements.txt` lists `crossref` but the code needs `crossrefapi`
  (fixed in the lock only).
- v2 data root `/media/mario/HDD/Data/NeuroScape_v2`:

  ```
  Manifests/                      v1 checksums (read-only)
  Internal/Base/V1/               copy of NeuroScape_original/Public/Data (read-only, verified)
  Internal/Delta/2025-10/HDF5/    VoyageLite, VoyageLarge, Domain (44,905 articles, no links)
  Internal/Delta/2025-10/CSV/     Neuroscience, Multidisciplinary CSVs of the Oct 2025 run
  Internal/Raw/CSV/               raw scrape shards (Neuroscience, Multidisciplinary)
  Internal/Reference/             Scimago (Neuroscience, Multidisciplinary), journal_lut.csv
  Internal/Intermediate/Models/   classifier + domain-embedding model (identical to v1 public models)
  Internal/Checkpoints/           scraped_articles.json, embedded_articles.json
  ```

- Run v2 steps with `BASEPATH` overriding `.env` (`load_dotenv` does not override existing variables):
  `BASEPATH=/media/mario/HDD/Data/NeuroScape_v2 PYTHONPATH=. conda run -n neuroscape_v2 python scripts/...`
  (from the repo root; scripts import `src.*`). Long runs use wrapper scripts and logs in
  `NeuroScape_v2/Internal/Logs/`.
- Scimago 2025 lists downloaded manually (Cloudflare blocks scripts). The 2025 export has no `Areas`
  column; `scraping.py` takes it from the most recent earlier list (all 2025 Q1 journals are covered).

### Phase 1 — started 2026-10-06 16:25

- `scraping.py` now honours `--start_year/--end_year`, defaulting to 1999–2023 (the v1 range). Also
  fixed: stale `pubmed_ids`/`metadata` after failed retries (could attach the previous article's metadata
  to a PMID), and the final partial shard was never saved.
- Run: `Internal/Logs/run_phase1_scrape.sh`, which runs Neuroscience then Multidisciplinary for
  2024–2025, one after the other because they share a checkpoint file. Log: `Internal/Logs/phase1_scrape.log`.
- Note: `max_results = 5000` per journal-year, as in v1. Very large multidisciplinary journals are
  capped at 5000 articles per year (same as v1).
- Progress 16:54: Neuroscience 2024 top-up finished in about 6 min. 40,883 PMIDs checked, only 82 new,
  so the Sep 2025 scrape was essentially complete. Rate is about 7,000 PMIDs/h; Neuroscience 2025
  expected done around 22:00, Multidisciplinary around midday on 2026-10-07.
- **Finding:** about 21% of newly scraped rows have `Year` 2026 (online in 2025, issue dated 2026).
  `year_cutoff = 2025` drops them, consistent with v1. **Decided 2026-10-06: drop** (they enter with the next update).
- **Finding:** `habanero.counts.citation_count` (CrossRef OpenURL) fails for every DOI, so `Citations`
  is NaN in all raw shards since Sep 2025, including the 2025-10 delta. Harmless because Phase 7 refetches
  all counts, but Phase 7 must use the REST API (`api.crossref.org/works/{doi}` → `is-referenced-by-count`).

### Phase 2–3 — code ready, tested 2026-10-06

- `merge_and_clean.py --year_cutoff` overrides the config value (the default is unchanged).
- New `scripts/ingestion/remove_known_articles.py` drops articles already in Base/V1 or the delta
  (matched by PMID or DOI). It keeps the complete dataframe as `articles_merged_cleaned_all.csv`.
- Articles that were embedded and rejected by the classifier in Oct 2025 (multi-area journals such as
  Medicine / Neuroscience) are skipped by `initial_embedding.py` via its checkpoint. They have no
  embedding shard in v2 and drop out again in `filter_disciplines.py`. They are not paid for twice.
- In the v2 root the existing scripts write to empty `Internal/Intermediate/*` directories, so their
  in-place rewrites only touch new articles.
- Smoke test on a sample (scratch root): 484 new articles, 337 kept. PMIDs are identical across
  CSV/lite/large/domain. Lite and large embeddings match their models (cosine 1.0). The domain model
  reproduces the stored delta domain embeddings exactly (max abs diff 0.0).
- Run after the scrape: `Internal/Logs/run_phase2_3.sh` (log to `Internal/Logs/phase2_3.log`).
