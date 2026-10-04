# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Stochastic Pirates  
**Team Members:** Gaurav Rajesh Patel (leader); Shivanshu jaiswal; Saransh samar; Aryan Tiwari  
**Institution:** Pandit Deendayaal Energy University  
**Submission Date:** 2-10-2026

---

## 1. Executive Summary

We use a local-only, supervised entity-resolution pipeline tailored to the challenge's precision-heavy macro F0.5 metric. The system first generates a bounded set of plausible Source-2/Source-3 candidates for every Source-1 record, then scores the candidates with one LightGBM binary classifier trained on the supplied ground truth and candidate-derived hard negatives. Candidate generation and scoring are partitioned by the observed country string, which keeps the method open to France and any future country labels without hard-coded country categories. No external databases, APIs, geocoders, web searches, or non-provided business data are used.



## 2. Methodology

Each name and address is canonicalized before comparison: Unicode is transliterated to a Latin representation; casing, punctuation, URL fragments, and common web-domain suffixes are removed; common legal business suffixes are excluded from the business-name core; and frequent address abbreviations are expanded. Numeric address components are normalized by converting values such as `0034` to `34` while retaining the first address number separately.

The pipeline creates one deterministic entity-level validation partition from Source-1 IDs using a fixed BLAKE2b hash. Thus, all matches belonging to a held-out Source-1 entity remain held out, avoiding pair-level leakage. It trains on a capped, deterministic sample of the remaining Source-1 records. Positive links come from `train_ground_truth.tsv`; negatives are hard look-alikes retrieved by the same blocker that is used during inference. The final model is a single LightGBM gradient-boosted tree classifier, seeded for reproducibility.

After training, separate Source-2 and Source-3 probability thresholds are selected using a grid search that directly maximizes macro F0.5 on the held-out entities. This explicitly treats singleton correctness and false merges according to the competition metric.

## 3. Candidate Generation / Blocking Strategy

Records are only compared inside the same observed country partition. For each target record the pipeline builds an inverted index of several complementary keys:

- Canonical compact business name, legal-suffix-stripped core name, first two name tokens, and sorted name tokens.
- Significant individual name tokens.
- Street/door number combined with rare address tokens or the final address token.
- First business-name token combined with an address number or location token.
- Pairs of rare address tokens.

Candidate records are ranked by the sum of inverse-frequency weights for their shared keys, rather than by a simple shared-key count. This makes several rare, independent agreements stronger evidence than a single common token. Extremely broad one-token/address postings are suppressed only for their weak key type; exact and composite keys are retained. The top 50 candidates per Source-1 record are scored by the model. The exact scored candidates are written to `candidate_pairs.tsv`, and matches are selected only from that list.

## 4. Model Architecture and Feature Engineering

The model is one LightGBM binary classifier (650 trees, learning rate 0.035, 63 leaves, row/feature subsampling 0.85). Its 26 numeric features include:

- Exact and fuzzy name agreement: normalized Levenshtein, Jaro-Winkler, token-sort/token-set ratios, token overlap, name-length ratio, character-trigram Jaccard, and first-token similarity.
- Address agreement: exact, edit, Jaro-Winkler, token-sort/token-set, and address-token Jaccard measures.
- Structural checks: address presence, shared address numbers, number conflict, primary-number agreement/conflict, exact core name with a missing target address, and a name-by-address interaction.
- Candidate context: the summed inverse-frequency blocking score and a Source-2 indicator.

Target exclusivity is not assumed blindly. The training labels are inspected first. Greedy highest-probability target assignment is enabled only if no target ID is linked to multiple Source-1 entities in the supplied training ground truth; otherwise all threshold-qualified links are retained.

## 5. Reproduction and Outputs

Install the pinned dependencies and run the command in `code/business_entity_resolution/README.md`. It regenerates both required TSV files from the supplied data, writes the trained model and metadata under `work/`, and records deterministic validation metrics in `work/validation_metrics.json`.

Before upload, run the organizer-provided `utils/validate_submission.py` against both TSVs. The archive must be rebuilt only after this validator reports `PASS`, and its `output/` directory must contain the final `matching_results.tsv` and `candidate_pairs.tsv` generated by that run.
