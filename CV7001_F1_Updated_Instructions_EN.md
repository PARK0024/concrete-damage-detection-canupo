# CV7001 — Image-Based Damage F1 Evaluation: Windows Instructions

This guide is for the project **Roughness-based CANUPO (SVM) for Concrete Surface Damage Detection**. It explains how to reproduce the F1 comparison between **CANUPO Only** and **Roughness–CANUPO (R–C)** without renaming your image files.

## 1. Prepare the files

Place all six program files together in the `concrete-damage-detection-canupo` root folder:

```text
CV7001_evaluate_f1.py
CV7001_F1_study_config_PRIVATE.json
CV7001_run_f1_windows.bat
requirements.txt
README.md
CV7001_F1_Updated_Instructions_EN.md
```

Keep the 12 input images in `CV7001_F1_input/`:

```text
CV7001_F1_input/
  S01.jpg          S01_CANUPO.jpg          S01_RC.jpg
  S02.jpg          S02_CANUPO.jpg          S02_RC.jpg
  S03.jpg          S03_CANUPO.jpg          S03_RC.jpg
  S04.jpg          S04_CANUPO.jpg          S04_RC.jpg
```

Other actual extensions (e.g. PNG) are accepted. The file **stems** must match; extensions do not need renaming. Use raw CANUPO rendering images, not colour-enhanced versions made for figures.

## 2. Install dependencies

Open File Explorer in the repository folder, click the address bar and enter `powershell`. Then execute:

```powershell
py -3 -m pip install -r requirements.txt
```

If the Python Launcher is unavailable, use `python -m pip install -r requirements.txt` instead. Install Python 3.10 or newer if it is not present.

## 3. Run the evaluation

From PowerShell (in the repository folder):

```powershell
.\CV7001_run_f1_windows.bat
```

**PowerShell requires the `./` or `.\` prefix** to invoke a batch file in the current folder. Alternatively, double-click the `.bat` file in File Explorer.

The Python equivalent is:

```powershell
py -3 .\CV7001_evaluate_f1.py --config .\CV7001_F1_study_config_PRIVATE.json --input-dir .\CV7001_F1_input --output-dir .\CV7001_F1_results --label-mode damage --no-sensitivity
```

To generate optional parameter-sensitivity results, omit `--no-sensitivity`.

## 4. Interpret the outputs

The automatically created `CV7001_F1_results/` folder includes:

- `f1_comparison.csv`: F1-score (%) of both workflows and the difference in percentage points.
- `per_model_metrics.csv`: TP, FP, FN, TN, precision, recall and F1.
- `S01_damage_review.png` through `S04_damage_review.png`: six-panel visual diagnostics.
- `S01_damage_alignment_check.png` through `S04_damage_alignment_check.png`: reference-image and scoring-ROI alignment diagnostics, using the **same exact scoring ground-truth mask** as the review.
- `S01_damage_GT_mask_rectified.png` through `S04_damage_GT_mask_rectified.png`: the exact 600×600 binary scoring masks (white = damage, black = non-damage).
- `*_damage_grid_labels.npz`: GT and predicted binary grid labels.
- `RUN_METADATA.json`: configuration checksum and methodological notes.

The full-photo panels show the **entire marked photograph** with its aspect ratio preserved and padding as needed; no photo area is cut off. **The review's GT panel and the alignment-check's rectified GT panel now visualise the very same GT mask used for F1.** The review inverse-projects it onto the full photo while the alignment check shows it directly in the 600×600 scoring frame. Both include the 3% crack-line buffer and the configured spalling polygons. As the two images use different perspectives, the shapes can look different even though they refer to the same mask. The bottom panels are the registered prediction grids.

### Why the photograph previously appeared excessively zoomed

The previous review visualisation showed the photo **rectified and cropped to a specimen-face ROI**, which could resemble an accidental enlargement. Also, its original ROI coordinates were recorded in the dimensions of an earlier annotated image; reading a differently sized copy without scaling those coordinates could crop the wrong portion of the image.

The updated program fixes both issues:

1. Review images show the full original photograph without cropping.
2. The evaluation ROI corners are rescaled automatically using the reference photo dimensions in JSON **when the same photograph is supplied at another pixel resolution**.
3. A separate `*_alignment_check.png` clearly displays the full photo with the scoring ROI outline and the rectified frame used in F1 calculations.

**Do not confuse preview correction with accuracy correction.** The full-photo panel is for display only; the F1-score is still calculated from the rectified specimen face. An actual change in photo crop, camera angle, or cloud viewport requires manual registration adjustment.

## 5. Verify the specimen-face ROI

For **each specimen**:

1. Open `S01_damage_alignment_check.png` (then S02, S03, S04).
2. In the first panel, verify the ROI outline corresponds to the concrete specimen face.
3. In the second panel, verify the scoring face is rectified as intended.
4. In the third panel, verify the magenta damage mask is in the correct positions.
5. If not, edit `photo_corners_xy` in `CV7001_F1_study_config_PRIVATE.json`. These coordinates are expressed in the **reference image size** specified by `photo_reference_size_wh`.
6. Re-run the batch file, and verify both models against the same reference mask.

The program prints a warning if the input photograph's aspect ratio differs from the stored reference by more than 5%. This can indicate a different crop/view; **do not use F1 percentages as final results until the ROI is checked**.

## 6. Evaluation settings

The fixed settings in JSON are:

| Setting | Value |
|:--|--:|
| Rectified scoring image | 600 × 600 pixels |
| Grid | 20 × 20 cells |
| Ground-truth positive threshold | 8% damage mask per cell |
| Annotation tolerance | 3% of face height |
| Minimum classified pixels | 15 per cell |
| Predicted damage threshold | 25% red (red/(red+blue)) |

The metric is `F1 = 2 TP / (2 TP + FP + FN)`. In S01, the damage reference comprises marked crack lines. In S02–S04, provisional spalling-interior polygons are added. This is **image-based region-level agreement**, not 3D point-cloud classification accuracy.

## 7. Compare with the report and submit to GitHub

The values in `f1_comparison.csv` depend on the **actual image files**. Prior report values may differ from your new run, especially if the images were recropped, re-rendered, compressed, or aligned differently. Match the final report table to the verified current CSV rather than copying older example percentages.

The repository should also contain the point-cloud input data, CANUPO `.prm` files and CloudCompare training/classification steps if the full workflow must be reproducible. The Python evaluator only reproduces the exported **image-level comparison**. Disclose CloudCompare/CANUPO usage and AI-assisted code/document preparation where required.
