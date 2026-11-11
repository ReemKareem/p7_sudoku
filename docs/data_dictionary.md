# Data dictionary

| Item | Real or generated | How |
|---|---|---|
| MNIST images (28x28, uint8 -> float 0-1) | **real** | Yann LeCun's MNIST; 60k train / 10k test, different writers |
| MNIST labels 0-9 | **real** | digit 0 dropped, labels 1-9 stored as 0-8 in `data/processed/mnist19.npz` |
| train / validation split | generated (seeded) | `make_data.py`: permute the 60k training images with seed 0, 10k validation, rest train, then drop digit 0; test set untouched |
| `x_train, y_train, x_val, y_val, x_test, y_test` | real | arrays in the processed file; `source` = "mnist" |
| Sudoku solution grid | generated | randomised backtracking, `random.Random(seed)` |
| Puzzle (givens) | generated | remove cells one at a time while the solution stays unique (25/30/35 givens) |
| Mosaic | generated | each given cell = a random MNIST **test** image of the right digit; blank cell = faint Gaussian noise (sigma 0.02) |
| Correctly-read pool | generated rule | test images the NN reads correctly (per digit) |
| Hard (misread) pool | generated rule | test images the NN misreads with the true class in its top-3 (if < 5 exist for a digit, lowest-p(true) images are added) |
| Injected misreads (level k) | generated | k random given cells use hard images, the rest use correctly-read images |
| Blank-cell detection | rule | fewer than 10 pixels brighter than 0.5 (not the NN) |
| NN probabilities | model output | softmax(logits / T), T fitted on the validation set (temperature scaling) |
| Correction cost | derived | max(0, ln p(read) - ln p(new)) |
| Experiment seeds | generated | puzzles: 1000+s (E1), 5000+10s+n (E2), 9000+s (E3); injection seeds 100s+level |
