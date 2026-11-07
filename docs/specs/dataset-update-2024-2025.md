# Spec: NeuroScape 2.0 — dataset update with 2024–2025

Status: **agreed** (spec interview 2026-10-06) · Supersedes the incremental-update part of `PLAN.md`

## 1. Goal

Publish **NeuroScape 2.0** on Zenodo (new version under the v1 concept DOI) covering
1999–2025 (**extended to 1980–2025 on 2026-11-03**, see §9 "Extension to 1980–1998"). Do all scraping and processing now (Oct 2026). In January 2027, refresh
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
- Neuroscience and Multidisciplinary journals, same Scimago quartile rule as v1: **Neuroscience Q1 + Q2,
  Multidisciplinary Q1** (per year's list; inferred from v1's composition, see §9).
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

**Phase 5 — Cluster assignment** (new script `scripts/clustering/assign_to_clusters.py`; outcome: kNN, see §9)
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
| `Added In` | `v1.0` (in v1), `v2.0` (2024–2025), `v2.0-late` (1999–2023 added in 2.0) or `v2.0-early` (1980–1998, proxy journal lists) |
| `Assignment Share`, `Assignment Margin` | Similarity-weighted kNN vote share of the assigned cluster, and its lead over the runner-up. Only for articles assigned in 2.0; v1 articles keep their Leiden labels (NaN) |
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

### Phase 5 — method decided and validated 2026-10-06

- Holdout validation: Leiden labels of v1 2023 articles vs. assignment using 1999–2022 articles only.
  - Nearest centroid (L2 mean): **0.765**; repo spherical centroids (`get_centroids`): 0.736. Both are
    below the 0.80 threshold, so per the spec we fall back to kNN.
  - kNN majority: k=10 0.917, k=25 0.947, **k=50 0.955**, k=100 0.936. Similarity-weighted k=50:
    **0.959**. Holding out 2015 instead gives 0.964.
- **Method:** similarity-weighted kNN vote, k = 50 (equals `graph_construction.num_neighbors`). Neighbours
  are v1 articles only, so the v1 clustering is the fixed reference and new articles don't influence
  each other. Config: `config/clustering.toml [assignment]`.
- The vote share is a good confidence measure. In the validation, articles with share ≥ 0.7 (66%)
  agree 1.000, 0.5–0.7 (19%) agree 0.956, and < 0.5 (15%) agree 0.790. The share and margin are released per article.
- Script: `scripts/clustering/assign_to_clusters.py` (`--validate` reruns the holdout). It assigns the
  2025-10 delta and the new articles from `Internal/Intermediate/HDF5/DomainEmbeddings` and writes
  `Internal/Intermediate/CSV/Neuroscience/articles_assigned.csv`.

### Phase 4 — code ready, dry-run 2026-10-06

- New `scripts/ingestion/fetch_new_links.py`. For every non-v1 article (the delta plus new ones) it stores
  raw candidates in `Internal/Intermediate/Links/link_candidates.jsonl`: citing PMIDs from PubMed
  `pubmed_pubmed_citedin` (batched, 200 per request) and reference DOIs from CrossRef (batched, 100 per
  request). Same sources as v1's `build_adjacencies.py`.
- Candidates are stored **unfiltered**. Intersecting them with the dataset and adding back-links to v1
  articles happen at assembly (Phase 6), so they stay valid if the set of articles changes.
  The script is resumable.
- **Decision (made while implementing):** reference DOIs are matched **case-insensitively**. 9.5% of v1
  DOIs contain uppercase letters, while CrossRef reference lists often spell them differently, so v1's
  exact matching missed some links. New links are therefore slightly more complete than v1's.
  v1 links are not recomputed (out of scope).
- Dry run: 300 delta articles in 36 s (about 8/s, so about 100K articles in 3–4 h). All 300 had
  references; median 48 references and 8 citing articles.
- Run: `Internal/Logs/run_phase4_5.sh` (link candidates, then cluster assignment) after Phases 2–3.

### Phase 7 — script ready, dry-run 2026-10-06

- New `scripts/ingestion/refresh_citations.py`. Fetches `is-referenced-by-count` from the CrossRef REST
  API (polite pool, 100 DOIs per request) for **all** articles (base, delta, new), and appends
  `Pmid, Doi, Citations, Fetched` to `Internal/Intermediate/Citations/citations_<tag>.jsonl`. Resumable.
- Dry run on 1,000 random articles took 11 s (so about 1.5 h for all ~530K). 1 DOI was not found in
  CrossRef. For v1 articles: median count 26 → 33, and 99.5% have count ≥ v1. An interrupted run
  (500/1000) resumed to exactly the same 1,000 articles with no duplicates.
- January: `Internal/Logs/run_phase7_january.sh` (citations for all, then link candidates refetched for
  2024–2025), then Phase 6 assembly with reference date 2027-01-01.
- Shared helpers: `src/utils/update.py` (dataset loading, DOI normalization, CrossRef batching, JSONL).

### Phase 6 — code ready, tested 2026-10-06

- New `scripts/ingestion/assemble_dataset.py`, `src/utils/assembly.py` and `config/ingestion/assembly.toml`
  (version, year range, `reference_date = 2027-01-01`, shard size). Writes to `Public/Data/` of the v2
  root (or `--output`):
  - `CSV/neuroscience_articles_1999-2025.csv`: v1 columns plus `Added In`, `Assignment Share`,
    `Assignment Margin` and `Citations Fetched`. v1 rows come first in v1 order, then added articles sorted
    by Year, Journal, Type.
  - `CSV/neuroscience_clusters_1999-2025.csv`: v1 text columns unchanged; updated `Size`,
    `Year First Article`, `MCR Research/Review`, Krackhardt indices and most cited/citing clusters.
  - `HDF5/DomainEmbeddings/shard_*.h5`: all articles, same order as the CSV, with updated links, age and
    citations. A missing citation count is stored as -1 in the HDF5 and NaN in the CSV.
  - `Graphs/article_citation.graphml` (rebuilt) and `Graphs/cluster_citation_density.graphml` (rebuilt);
    `article_similarity.graphml`, dimensions/trends CSVs and models are copied from v1.
  - `assembly_report.json` (counts per source and year, link totals, provisional flag).
- Modes: **provisional** (no `--citations`): v1 rows keep their v1 citation values, added articles have
  no counts. **Final** (`--citations citations_2027-01.jsonl`): all counts from the refresh, ages
  and rates relative to `reference_date`. Runner: `Internal/Logs/run_phase6.sh`
  (`CITATIONS=... ./run_phase6.sh` for final).
- Link handling: candidates are resolved against the full dataset (DOIs case-insensitive, self-links
  dropped). Later candidate files override earlier ones per article. Then in/out links are made symmetric
  across all articles (as `update_links` did in v1).
- **Density analysis reimplemented with sparse matrices.** On v1 inputs it reproduces the published v1
  values exactly: Krackhardt max abs diff 6e-16, most cited/citing clusters 175/175 identical, density
  weights max rel diff 5e-15. Runtime 21 s.
- **Finding (published v1 data):** `article_citation.graphml` in Zenodo 1.0.1 has 2,396,655 edges, but
  v1's out-links total 7,702,484. It contains the complete out-links of only 128,894 of the 415,393
  articles with out-links. All its edges are correct (citing → cited), the rest are missing. It was built in
  `results_0x_katz_ignore.ipynb` (likely interrupted or wrongly skipped articles). The paper figures and
  the density/Krackhardt statistics do not use it (they use the HDF5 links). The v2 graph is complete.
  **Resolved 2026-10-07, no 1.0.2:** the published Zenodo 1.0.1 (`NeuroScape_v101.zip`, md5
  `d6273d39…` verified against the record) never contained `article_citation.graphml`; it existed only
  locally. The complete v1 graph (7,702,484 edges; same nodes, order and cluster attributes; superset of
  the old edges) replaced it in `NeuroScape/Public` and `NeuroScape_original`. The old file is kept as
  `article_citation_incomplete.graphml`, and both manifests were regenerated (only these two files differ).
  `Internal/Base/V1` in the v2 root still holds the old copy; it is not used (assembly rebuilds the graph).
- Provisional test (v1 + 2025-10 delta + 300 dry-run link records; `Internal/Test/assembly_provisional`):
  - 506,221 articles (461,316 v1.0 / 26,721 v2.0 / 18,184 v2.0-late), no duplicate PMIDs/DOIs.
  - v1 rows identical to v1 in all v1 columns and order. HDF5 order matches the CSV.
  - Links symmetric (in = out = 7,707,798). All v1 links preserved; 4,836 v1 articles gained links.
  - Article graph edges = out-link total. Cluster sizes sum to the total, text columns unchanged.
  - A rerun into the same output works. Runtime about 15 min.
- Final-mode test (synthetic refresh covering all 506,221 articles; test outputs deleted afterwards):
  - Every article has a count and fetch date. `Citation Rate = Citations / Age` exactly.
  - Ages are measured to 2027-01-01: v1 ages shift by +1.99 years (IQR 0.004), so v1 ages were
    measured around January 2025.
  - The HDF5 and cluster MCRs use the refreshed values.

### Phases 2–5 — running since 2026-10-07 18:17

- The scrape finished cleanly: Neuroscience at 2026-10-06 23:04, Multidisciplinary at 2026-10-07 08:24
  (450 journal-years, no failures).
- `Internal/Logs/watch_and_run_phase2_5.sh` (log `watcher.log`) waits for the scrape, checks its log, then
  runs `run_phase2_3.sh` (log `phase2_3.log`) and `run_phase4_5.sh` (log `phase4_5.log`). It stops at the
  first failure and writes `=== ABORT ...` to `watcher.log`.
- After `remove_known_articles`, Neuroscience: 34,128 (2025) + 21,406 (2024) + older. Most of the 2024 and
  older ones were rejected by the classifier in Oct 2025 and are skipped by the embedding checkpoint; about
  34K articles get embedded.
- 18:17–20:08: Phases 2–3 completed on the Q1-only scrape (21,284 new articles kept: 20,676 from 2025 and
  608 from 2024).
- **Correction (2026-10-07): the quartile rule.** The 2025 count looked about 23% low against 2024. The cause:
  v1 used Scimago **Q1 + Q2** for Neuroscience journals (and Q1 only for Multidisciplinary). Evidence:
  v1 articles by that year's quartile were 2005 Q1 3,729 / Q2 2,916; 2015 Q1 11,527 / Q2 6,246; 2023 Q1 7,821 /
  Q2 9,813, while Multidisciplinary is Q1 only. *Brain Sciences* is present exactly in the years it was Q2
  and absent in its Q3 years. The Sep 2025 scrape (2024) also included Q2, but the Phase 1 run used the
  script default `--quartile Q1`, so the 2025 Neuroscience Q2 journals were missing. For journals present
  in both years, 2025 is at 96% of 2024. Fix: `Internal/Logs/run_q2_chain.sh` scrapes Neuroscience Q2 for
  2024–2025 (log `phase1_scrape_q2.log`), clears the derived new-article HDF5 sets, then reruns Phases 2–3 and
  4–5 (chain log `q2_chain.log`). Already embedded articles are skipped via checkpoint; the large-model
  re-embedding of about 21K articles costs < $1.
- **Fix: Phase 4 hang.** `fetch_new_links.py` stalled at 20:26 on a PubMed request without a timeout
  (8,500 articles done, resumable). `socket.setdefaulttimeout(120)` was added to `fetch_new_links.py` and
  `scraping.py`, so stalled requests fail and are retried. The watcher was stopped and replaced by the chain.

### Extension to 1980–1998 — decided 2026-11-03, scrape running

- **Decision:** v2 covers 1980–2025. Scimago starts in 1999, so 1980–1998 use **proxy journal lists**. The release
  carries a note that 1980–1984 abstract coverage is incomplete. 1970 was rejected because PubMed has essentially
  no abstracts before 1975.
- **Why a proxy is acceptable:** the quartile rule is fairly stable over time. Of Neuroscience journals in Q1+Q2
  in year *t* (and still listed), 93% are still in Q1+Q2 at *t*+1, 89% at +5, 86% at +10 and 81% at +20. For
  Multidisciplinary Q1 the figures are 85%, 75%, 67% and 61%; that list is small and the classifier filters it anyway.
- **Abstract coverage** (PubMed `hasabstract`, examples): J Neurophysiol 1970 0/66, 1975 97/101, 1980 77/184,
  1985 202/203; Brain Res 1975 379/928, 1980 743/1229, 1985 1835/1841. Coverage is patchy and varies by journal
  until about 1984, and essentially complete from 1985. Articles without an abstract are dropped by cleaning
  (lower word limit), as in v1.
- **Proxy rule** (new `scripts/ingestion/build_proxy_journal_lists.py`):
  - Core: journals meeting the v1 rule (Neuroscience Q1+Q2, Multidisciplinary Q1) in at least 3 of 1999–2003.
    The quartile is the modal in-rule quartile in those years.
  - Renamed journals: the NLM Catalog's `Preceding` titles are followed by ISSN (up to 3 renames back) and
    scraped under their MEDLINE abbreviation.
  - Each year's list keeps only journals published that year (NLM first and end years).
  - Output: `Internal/Reference/ScimagoProxy/<discipline>/`, one file per year in Scimago format, plus
    `proxy_journals_<discipline>.csv`.
  - Neuroscience: 173 core journals plus 29 preceding titles (e.g. Electroencephalogr Clin Neurophysiol →
    Clinical Neurophysiology, Paraplegia → Spinal Cord, Anat Embryol → Brain Structure and Function). The
    yearly lists hold 84 journals in 1980 and 168 in 1998.
  - Multidisciplinary: 14 core journals, no renames.
- **Known bias:** journals that ceased or declined before 1999 are missing (survivorship). The 1980–1984
  article counts are thinned unevenly by missing abstracts.
- **Volume** (PubMed counts for the proxy lists, capped at 5000 per journal-year):
  - Neuroscience: 1980 8,428 (5,270 with abstract), 1989 16,135 (14,705), 1998 25,122 (22,910).
  - Multidisciplinary: about 5,000–5,700 per year (3,200–4,100 with abstract).
  - In total about 410K PMIDs. At about 2 PMIDs/s the scrape takes about 2.5 days.
- **Code changes:**
  - `scraping.py --journal_lists scimago_proxy` (default `scimago` = v1 behaviour); input files are matched by
    the `scimagojr ` prefix. Directory key `scimago_proxy` added to `config/directories.toml`.
  - `assembly.toml`: `start_year = 1980` plus `notes`, which are written to `assembly_report.json`.
  - `assemble_dataset.py`: `Added In = v2.0-early` for Year < 1999, and added articles are filtered to
    `start_year ≤ Year ≤ end_year`.
- **Run** (started 2026-11-03 08:14): `Internal/Logs/run_early_scrape.sh` runs Neuroscience Q1, Neuroscience
  Q2 and Multidisciplinary Q1 one after another. Logs: `phase1_scrape_early_<discipline>_<quartile>.log`.
  The checkpoint before the run is backed up as `Checkpoints/scraped_articles.before_early.json`.
- **Next (manual, after the scrape):** as in `run_q2_chain.sh`, clear the derived new-article HDF5 sets, then
  run `run_phase2_3.sh` and `run_phase4_5.sh`, and rerun the Phase 6 provisional assembly. Then check:
  - the `v2.0-early` counts per year and journal (sanity notebook);
  - the `Assignment Share` distribution by year. Topics that faded before 1999 have no v1 cluster, so expect
    lower shares for the 1980s.
- **Scrape status 2026-11-05:**
  - Neuroscience Q1: done 2026-11-04 10:52; 182,051 PMIDs checked, 139,231 kept (with abstract).
  - Neuroscience Q2: done 2026-11-05 04:51; 128,911 checked, 88,310 kept. Every search in both Neuroscience
    runs returned a result.
  - **Multidisciplinary Q1 failed silently** (04:51–06:54). None of the 230 journal-year searches returned a
    result, probably because of NCBI errors at that time; the same query works afterwards. The cause is in
    `get_id_list`: it returned `None` on an unreadable response instead of raising, so the retry loop in
    `scraping.py` stopped at once and skipped the journal-year without logging anything.
  - **Fix:** `get_id_list` now raises, so failed searches are retried (`num_attempts`, `sleep_time`), and
    `scraping.py` logs `Search failed after N attempts` for any that still fail. Successful searches behave
    as before.
  - Rerun: `Internal/Logs/run_early_scrape_multi_retry.sh`, started 2026-11-05 19:5x. The failed log is kept as
    `phase1_scrape_early_Multidisciplinary_Q1.failed_ncbi.log`. No PMIDs had been checkpointed, so the rerun
    is clean.
- **Phases 2–3 on the extended set** (2026-11-06 18:17 – 2026-11-07 01:53, after the three derived HDF5 sets were cleared):
  - New articles after cleaning and removing known ones: Neuroscience 308,307; Multidisciplinary 210,815. The
    Multidisciplinary figure includes earlier-rejected 1999–2025 articles, whose lite embeddings come from the
    checkpoint.
  - After the classifier: 915 shards (about 183K articles), each with lite, large and domain embeddings.
- **Phase 4 failure (2026-11-07 02:10) and fix:** CrossRef returned `HTTP 431 Request Header Fields Too Large` at
  batch 88 of 1,486. The batch filter puts 100 DOIs into the URL, and long 1990s DOIs (e.g. Wiley SICI) push it
  over the limit.
  - Fix: `fetch_crossref_works` now splits the batch in half recursively on HTTP 414/431, and `crossref_request`
    no longer retries those errors. This also protects the Phase 7 refresh.
  - The failed batch then went through (resumed at 87,877 processed). Phases 4–5 restarted; the failed log is
    `phase4_5_early.failed_431.log`.
- **Phase 4 done** 2026-11-07 11:33. 1,398 batches with no failures; 9 `IncompleteRead` errors each succeeded on retry.
- **Phase 5 failure and environment fix:**
  - `assign_to_clusters.py` failed on import: `faiss-cpu` 1.8.0 is built for NumPy 1.x, but the env had NumPy
    2.5.3. Matplotlib and seaborn had been pip-installed into `neuroscape_v2` on 2026-10-08 15:53, which pulled in
    NumPy 2.5.3 and contourpy 1.4.
  - Fix: NumPy restored to the lock version 1.26.4 and contourpy set to 1.3.3; matplotlib and seaborn still
    import. Phases 2–3 had already run under NumPy 2.5.3; this has no effect on the embeddings (Voyage API and torch).
- **Phase 5 done** 2026-11-07 13:42. 227,706 added articles assigned (44,905 delta + 182,801 new); median vote share
  0.796.
  - By period (n, median share, share < 0.5):

    | Period | n | Median share | Share < 0.5 |
    |---|---|---|---|
    | 1980–84 | 17,493 | 0.701 | 27.4% |
    | 1985–89 | 31,599 | 0.723 | 25.2% |
    | 1990–94 | 50,485 | 0.764 | 22.4% |
    | 1995–98 | 48,952 | 0.821 | 18.1% |
    | 2024–25 | 34,272 | 0.840 | 16.4% |

    As expected, the fit to the v1 clusters falls with distance from 1999, but only moderately.
  - 148,529 articles from 1980–1998 survive the classifier, rising from 2,356 (1980) to 13,133 (1998). For
    comparison, v1 has 12,404 in 1999, so the series is continuous across 1998/1999.
