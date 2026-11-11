Data source: **mnist** (test set, digits 1-9 only, n=9020)

| Model | Test accuracy | NLL | ECE (15 bins) | Temperature |
|---|---|---|---|---|
| Logistic regression | 0.9218 | 0.2727 | 0.0203 | 1.000 |
| NN (uncalibrated) | 0.9772 | 0.0756 | 0.0072 | 1.000 |
| NN + temperature scaling | 0.9772 | 0.0718 | 0.0036 | 1.387 |
