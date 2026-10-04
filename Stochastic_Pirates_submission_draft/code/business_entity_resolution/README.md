# Stochastic Pirates — Business Entity Resolution

This is a self-contained, local-only pipeline for the Amazon ML Challenge. It reads only the supplied TSV files; it makes no network calls and uses no external business data.

## Setup

Use Python 3.10 or 3.11 and install the exact dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the outputs

Run this command from `code/business_entity_resolution` after placing the challenge data in its normal `dataset/train` and `dataset/test` locations (or pass absolute paths):

```powershell
python src/pipeline.py run `
  --train-dir ..\..\..\dataset\train `
  --test-dir ..\..\..\dataset\test `
  --output-dir ..\..\..\output `
  --work-dir ..\..\..\work
```

The command performs a deterministic entity-level validation split, trains one LightGBM classifier, selects separate S2/S3 decision thresholds to maximize validation macro F0.5, and writes:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `output/validation_metrics.json`

It also saves `work/model.txt` and `work/model_metadata.json`, allowing prediction-only reruns:

```powershell
python src/pipeline.py predict `
  --test-dir ..\..\..\dataset\test `
  --output-dir ..\..\..\output `
  --work-dir ..\..\..\work
```

## Design notes

The program partitions records by the open-set `country` field, builds IDF-ranked candidate blocks from canonical names, address tokens, numeric address anchors, and name/address combinations, and scores only those candidates. `candidate_pairs.tsv` is written from the exact candidate list scored by the model, so every predicted match is necessarily a candidate. The model is trained on true pairs plus candidate-derived hard negatives. A target may be assigned to only one source-1 entity only if that exclusivity is verified in the training ground truth; this avoids relying on an undocumented assumption.

## Validation

Run the organizer-supplied validator after prediction:

```powershell
python ..\..\..\utils\validate_submission.py `
  --matching ..\..\..\output\matching_results.tsv `
  --candidate ..\..\..\output\candidate_pairs.tsv `
  --test-dir ..\..\..\dataset\test
```
