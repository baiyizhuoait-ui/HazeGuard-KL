# 26-grid repository audit and safe cleanup plan

Audit baseline: remote `origin/main` = `3cdf1ba3a868361406dcf8d9158a482d4da4bc72`; working branch observed as `fix/02-26grid-final`. Read-only audit; this task changed no repository file, branch or object. User confirmation of 26-grid scope is the new intended acceptance direction; current repository findings below refer to the pre-edit baseline.

## Decision before edits

**Delete zero tracked files.** All 42 tracked files have a necessary executable, accepted-output, provenance, interface, dependency, collaboration or course role. There are no byte-identical tracked duplicate files. The two parquet files are input and derived output, not duplicates. The two Agent 2 dependency files serve human-readable direct pins and the complete reproducible environment lock; neither is redundant with upstream `requirements.txt`.

Only ignored local test/bytecode caches are cleanup candidates. History/branches remain untouched; no force push, history rewrite, LFS purge or branch deletion is warranted. Formalizing the accepted 26-grid contract and removing active old-eight selection logic is more useful than deleting valid evidence.

## Search for a latest independent technical specification

Commands inspected all tracked paths, all fetched remote branch trees and full reachable path history for `.docx`, `.pdf`, `spec` and `规范` names. No independent current 26-grid technical specification file was found. No tracked DOCX/PDF exists in any fetched branch, nor any such path in reachable Git history. `src/config.py` mentions v2.1 in quality-control comments, but this is not evidence of an independent v2.1 specification document.

The current 26-grid implementation contract is distributed across these evidence sources:

- `README.md:67` establishes 26 deduplicated CAMS grid cells and the 3/10/13 role split; `README.md:190` gives the real Agent 2 table contract.
- `src/config.py:82` starts the 26-node registry, `src/config.py:123` derives node/receptor IDs and coordinate mapping. This is the machine-readable authoritative registry.
- `docs/node_discovery.json` preserves the 33-candidate to 26-grid discovery evidence; `tools/discover_nodes.py` is the historical discovery procedure.
- `data/data_dictionary.json` and `data/manifest.json` define the real 35-column input and checksum/schema evidence.
- `docs/handover/01_data_and_baseline.md:127` records the original-eight collapse; member 1 handover and Section 1 remain owned historical evidence.
- `models/metrics_stage2_features.json` defines exact feature names, model input allowlist, missingness, eligibility and source/output hashes; member 2 report/handover documents computation and limits.

Recommended addition: a clearly dated current 26-grid specification/acceptance document with explicit source evidence and user-confirmed scope, linked by README and the member 2 handover. Describe it as the newly consolidated current contract; do not claim to have discovered an absent newer formal document or invent a version provenance. Keep the old specification references as historical change rationale, clearly superseded for current acceptance.

## A/B/C/D/E classification

A = required current artifact/support; B = historical/provenance evidence; C = reproducible local cache; D = exact redundant content candidate; E = downstream input/interface dependency. A file may have two roles. Categories are decisions about retention, not moving files into folders.

| Tracked file | Class | Retention reason |
|---|---|---|
| `.gitattributes` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `.gitignore` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `CONTRIBUTING.md` | A | Project navigation, contracts and course method scope; update stale navigation/documentation carefully |
| `README.md` | A | Project navigation, contracts and course method scope; update stale navigation/documentation carefully |
| `data/data_dictionary.json` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `data/feat_spatiotemporal.parquet` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `data/haze_panel_clean.parquet` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `data/manifest.json` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `docs/handover/01_data_and_baseline.md` | B/E | Owned Agent 1 report/handover; retained scientific provenance and downstream comparison |
| `docs/handover/02_eda_and_spatiotemporal.md` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `docs/node_discovery.json` | B/A | Candidate deduplication evidence/tool; historical evidence remains necessary; retain |
| `docs/syllabus_coverage.md` | A | Project navigation, contracts and course method scope; update stale navigation/documentation carefully |
| `figures/fig_baseline_dt_confusion.png` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `figures/fig_baseline_dt_importance.png` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `figures/fig_eda_class_imbalance.png` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `figures/fig_eda_correlation_matrix.png` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `figures/fig_eda_diurnal_seasonal.png` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `figures/fig_eda_spatial_crosscorr.png` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `figures/fig_eda_wind_rose_pollution.png` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `models/baseline_dt.joblib` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `models/cleaning_report.json` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `models/gate0_report.json` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `models/metrics_stage1.json` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `models/metrics_stage2_eda.json` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `models/metrics_stage2_features.json` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `models/metrics_stage2_verification.json` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `reports/sec1_data_and_baseline.md` | B/E | Owned Agent 1 report/handover; retained scientific provenance and downstream comparison |
| `reports/sec2_3_and_2_4_1_agent2.md` | A/E | Agent 2 accepted delivery artifact/contract consumed by member 3; regenerate only with matching audit |
| `requirements-agent2.lock.txt` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `requirements-agent2.txt` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `requirements.txt` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `src/01_baseline_dt.py` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `src/01_data_pipeline.py` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `src/01_verify_data.py` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `src/02_eda_analysis.py` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `src/02_spatiotemporal_features.py` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `src/config.py` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `src/utils_metrics.py` | A/E | Agent 1 authoritative input, baseline, schema or evaluation dependency; preserve bytes/path |
| `tests/test_agent2.py` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `tools/check_prose.py` | A | Executable/generation/test/dependency/repository support; retain complementary roles |
| `tools/discover_nodes.py` | B/A | Candidate deduplication evidence/tool; historical evidence remains necessary; retain |
| `tools/verify_agent2.py` | A | Executable/generation/test/dependency/repository support; retain complementary roles |

All 42 baseline tracked files are covered above. **C: zero tracked files. D: zero tracked files.** Baseline checked-out bytes total 108,158,863 including the expanded LFS parquet; Git stores its 133-byte pointer plus the LFS object separately.

Observed ignored caches, optional removal only after active test/generation processes finish:

| Directory | Files | Bytes | Decision |
|---|---:|---:|---|
| `.pytest_cache/` | 5 | 3,509 | C; disposable local pytest metadata |
| `src/__pycache__/` | 3 | 73,167 | C; disposable bytecode |
| `tests/__pycache__/` | 2 | 141,367 | C; disposable bytecode |

`data/raw/` is ignored by `.gitignore:8` and was absent locally. It would be C raw-response cache, but preserve it until any upstream data/version provenance need is resolved; it is not required to regenerate Agent 2 from supplied cleaned parquet. No untracked nonignored project files were observed. Exclude caches from delivery ZIP without deleting required output files. If actually removing caches later, resolve and verify each absolute directory stays under the checkout, then use PowerShell `Remove-Item -LiteralPath`; do not use `git clean -fdx` or recursive deletion against inferred locations.

## Old-eight references and stale navigation

| Evidence at baseline | Meaning | Recommended concrete edit |
|---|---|---|
| `src/02_spatiotemporal_features.py:55` falls back to `data/kl_8stations_clean.parquet` | Active legacy input selection, although current valid file wins | Default only to current `haze_panel_clean.parquet`; require explicit input for alternate data and validate adopted registry. Keep a clear missing-input error rather than a misleading old path. |
| `src/02_eda_analysis.py:223-228` first builds old `Port_Klang/Banting -> Cheras_KL/XMUM_Sepang` pairs | Current 26-grid execution falls back, but old names still control requested/missing status | Select current receptor comparison pairs directly; registry/role-validated `Port_Klang -> KL_City/Sepang`. Expose old mapping as historical context only; do not alias unavailable series. |
| Member 2 report opening and old-v2 comparison table | Accurately documented an earlier mismatch, but current formal 26-grid adoption changes acceptance language | Make adopted 26-grid current scope explicit; retain former-eight history instead of presenting it as an outstanding deliverable failure. |
| `docs/handover/02_eda_and_spatiotemporal.md:71` documents old-path fallback | Must track changed runtime default | Update default-path documentation and explicit alternate-input policy. |
| `README.md:163` says member 3 is waiting for 02 | Stale now that 02 merged | State 02 ready; 03 implementation not yet supplied. Do not claim 03 completed. |
| README directory tree at 214 onward lists only 01 products | Incomplete navigation | Add actual 02 scripts, featured parquet, metrics, tests, standalone dependencies and current specification. |
| `CONTRIBUTING.md:75-77` commit example asserts diurnal peaks, monsoon 1.8x and 42 new columns | Illustrative old text can be mistaken for real findings | Replace with neutral example or actual audited 56-column family counts; no fixed expected physical pattern. |
| `CONTRIBUTING.md:101,143` uses `03_fuzzy_granulation.py` but ownership table specifies `03_granulation_features.py` | Member 3 filename contract inconsistency | Align prose/example with ownership table; no member 3 code exists to rename. |
| `CONTRIBUTING.md:128` and member 1 handover cite 78,100 receptor training rows | Historical rounded/stale count | Current measured raw count 78,102 and audited eligible count 77,753; update current contract while preserving member 1 historical artifact or adding correction link. |
| `CONTRIBUTING.md:193,204,206` says every file <50 MB and current size 26 MB | Conflicts with accepted 80.6 MB parquet delivered through Git LFS | Document LFS object exception plus actual packaging sizes/checksums; keep ordinary Git binary size limits. |
| `src/config.py:93` has Banting in Sepang members; historical discovery tool starts from original candidates | Valid candidate deduplication provenance | Retain. This does not create a separate Banting time series. |

Do not remove every occurrence of “eight”, Banting or v2.0 mechanically; distinguish active defaults from evidence explaining why the adopted data contains 26 cells. No hard-coded array assumption of eight rows/stations was found in the current feature generator. Avoid modifying member 1 data/config/cleaning/model code as repository “cleanup”; those are downstream dependencies.

## Branch/history audit and member 3 status

| Fetched remote ref | SHA | Main commits ahead / ref commits ahead | Actual content |
|---|---|---|---|
| `origin/main` | `3cdf1ba3a868361406dcf8d9158a482d4da4bc72` | 0 / 0 | 42 files, merged Agent 2 |
| `origin/feat/02-eda` | `998515cf630ff8f786b8a633e22ca4668dd9b35f` | 4 / 0 | Historical 40-file Agent 2 delivery with earlier artifacts |
| `origin/fix/02-agent2-verified` | `61ffbd7a652a10c43eb528811397d1dd5ab014bb` | 1 / 0 | Accepted 42-file fix branch, merged through PR #1 |
| `origin/feat/03-granulation` | `8d8c59c7e89d9323b2dbb382c358a9c6b222d6bc` | 7 / 0 | 23-file old scaffold, no member 3 implementation |
| `origin/feat/04-nominal-rf` | same `8d8c59c...` | 7 / 0 | Reserved downstream scaffold |
| `origin/feat/05-ordinal-rf` | same `8d8c59c...` | 7 / 0 | Reserved downstream scaffold |
| `origin/feat/06-cost-explain` | same `8d8c59c...` | 7 / 0 | Reserved downstream scaffold |
| `origin/feat/07-streamlit` | same `8d8c59c...` | 7 / 0 | Reserved downstream scaffold |

Every fetched remote branch was checked with `git ls-tree -r`. There is no `src/03_*`, `data/features_final.parquet`, `models/selected_features.json`, `fig_granulation_bands.png` or `fig_feature_ablation.png` on any fetched branch, and no corresponding historical path was found. Member 3 currently has an agreed downstream file interface and reserved branch, not implemented code. Preserve `data/feat_spatiotemporal.parquet`, the feature audit allowlist, original labels/splits/IDs and all columns.

Do not delete any historical or downstream branch. Their zero ahead count makes them technically ancestors, but other members may retain local work or expect these reserved refs. Report that branches 03–07 need a normal update from main before work; choose merges/rebases according to each owner's unpushed-work state. Do not force-update their remote branches. Local `main` was still `2cd57c5` at first inspection while working branch tracked `origin/main`; this is local ref bookkeeping, not remote regression.

## Executable safe plan for root

1. Save this pre-edit audit (done). Record origin/main SHA and clean branch state before mutation.
2. Add current 26-grid contract and acceptance checklist referencing real registry, schemas and measured statistics; link README and handover. State 3 receptors/10 corridors/13 sources, actual data range, model population, input allowlist and offline-audit restrictions.
3. In member 2 owned files, remove active old-eight input/pair selection assumptions, preserving explicit alternate-input semantics only if intentionally documented. Keep Agent 1 source/data/models unchanged.
4. Update stale current navigation/status and contradictory member 3 filename examples/LFS packaging guidance. Treat member 1 handover as historical evidence, linking a correction instead of silently changing its results.
5. Delete no tracked files or refs. Optional cache cleanup is limited to verified ignored cache directories after tests. No LFS object pruning, archived ZIP removal, remote branch deletion, `reset`, force push or whole-tree move.
6. Regenerate 02 outputs on supplied input; verify five plots, feature-family counts, exact original-field preservation, no-NaN-as-zero fabrication, source/output hashes and perturbation tests. Current accepted contract has 56 added columns: 24 lag, 9 rolling, 2 change/slope, 6 cyclical, 2 wind, 7 spatial/availability, 6 audit-only flags.
7. Verify member 3 read snippet using `model_feature_columns`, `role == 'receptor'`, matching original split/labels, and eligible offline evaluation rows. Preserve all 35 original input fields. No claim that nonexistent 03 code was executed.
8. Review tracked diff: removals should be empty; permitted content edits must match scope; input SHA remains `97b5db123352861b6cd41224daf1ce278c90d6d528031b9bdc2744317a5ead93`.
9. Use isolated branch and PR, respect protected-main rules, fetch merged remote main anew, then create a complete LFS-expanded delivery ZIP with actual data and recorded commit. Validate from extraction without `.git` in the locked environment. Distinguish repository tracked additions from packaging-only metadata.

No cleanup action or new remote mutation was performed by this audit subtask.
