# Methodology and limitations

## Experimental conditions

1. **Original exploratory set:** 20 scenarios (normal ID 0 plus 19 labeled malfunction scenarios), 12,152 records and 96 sensor features. Fault time-zero samples were dropped. Fault labels indicate scenario source, not confirmed onset.
2. **Revised experimental set:** scenario IDs 5, 8, 14 excluded after matching measurements were found to coincide with normal data (10,339 rows, 17 scenarios). These excluded scenarios remain in the original prepared set for demonstration stress tests.
3. **Initial chronological split:** 80% early readings / 20% later readings per scenario. Readings from the same run are correlated; these results may overestimate generalization.
4. **Revised chronological 70/15/15 split:** Used for exploratory configuration checks. Many configurations reached 100% on this comparatively easy setup; this is not independent evidence of perfection.
5. **Scenario-grouped four-fold experiment:** 16 retained malfunction scenario IDs were divided across folds. All folds used early normal samples (time <400) for training and later normal samples for testing. This evaluates unseen *fault scenario labels*, but it does not evaluate unseen *normal runs*.
6. **Feature ablation:** within each fold, XGBoost importance was calculated on fold-training data before choosing 5, 15, 30 or 96 features. Feature sets can differ between folds.
7. **Statistical comparison:** one-sided exact Wilcoxon over four RF/XGBoost fold pairs, p = 0.0625. Due to only four correlated fold results, this is exploratory and not confirmatory.

## Major limitations

- High class imbalance and only one normal-operation simulation.
- At aligned simulation times, IDs 5, 8, 14 matched normal sensor measurements; labels are therefore observationally indistinguishable using the recorded inputs.
- Some early simulation records have contradictory labels for identical sensor values.
- Normal test period repeated across the four scenario-CV folds, so observations are dependent.
- The test scenarios have informed exploratory analysis; final unbiased model selection would require new independent data.
- Numerical model score is not calibrated risk or guaranteed fault probability.
- Some sensor names lack unit documentation in the provided work and should not be assigned guessed units.
- The saved application model is fit using the full revised dataset; in-app examples cannot be used to claim generalization.

## Recommendations

Collect more independent normal runs and repeated fault runs; create scenario/run-grouped train, validation, and locked test sets; verify event timestamps and labels; calibrate probabilities on independent validation data; investigate drift and detection delay; and evaluate operator-facing alert costs under realistic constraints.
