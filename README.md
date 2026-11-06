# Roughness-based CANUPO (SVM) for Concrete Surface Damage Detection

CV7001 Machine Learning Mini-Project — data and reproducible **image-based F1 evaluation** of CANUPO Only versus roughness-assisted CANUPO (R–C).

## 1. Project overview

Four concrete specimens (S01–S04) were scanned with an iPhone 16 Pro LiDAR sensor at the NTU Construction Laboratory. CloudCompare v2.13.2 and its CANUPO plug-in were used to classify damage (cracks and, where applicable, spalling) versus intact concrete. The direct CANUPO classification is the baseline; the R–C method applies local roughness filtering before CANUPO classification.

| Specimen | Selected roughness LNR (m) | Damage evaluated |
|:--|--:|:--|
| S01 | 0.025 | Primarily cracking |
| S02 | 0.010 | Cracking and spalling |
| S03 | 0.015501 | Cracking and spalling |
| S04 | 0.030 | Cracking and spalling |

The local neighbourhood radius (LNR) defines the **roughness search radius**, not the SVM kernel parameter. The CANUPO training and inference were run in CloudCompare; the Python program in this repository evaluates the exported classification images, not a standalone SVM implementation.

## 2. Folder layout

```text
concrete-damage-detection-canupo/
  CV7001_evaluate_f1.py
  CV7001_F1_study_config_PRIVATE.json
  CV7001_run_f1_windows.bat
  requirements.txt
  README.md
  CV7001_F1_Updated_Instructions_EN.md
  CANUPO_Procedure.md                 # opening and reusing saved .bin/.prm projects
  CV7001_F1_input/
    S01.jpg       S01_CANUPO.jpg       S01_RC.jpg
    S02.jpg       S02_CANUPO.jpg       S02_RC.jpg
    S03.jpg       S03_CANUPO.jpg       S03_RC.jpg
    S04.jpg       S04_CANUPO.jpg       S04_RC.jpg
  CV7001_CANUPO_Data Set/           # point clouds, classification outputs, .prm files
  CV7001_Original Scan Data set/    # original scan exports
  Appendix_Data Analysis.xlsx       # LNR experiments and results, if included
  CV7001_F1_results/               # auto-created evaluation outputs
```

The image extensions above are **examples**. The program recognises `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` and `.webp`; no filename changes are required when the stems match `S01`, `S01_CANUPO`, `S01_RC`, etc. Windows may hide extensions in File Explorer.

`CV7001_F1_study_config_PRIVATE.json` retains its historical filename for compatibility with the batch launcher; it is simply the configuration used for the study.

## 3. Install and run (Windows)

Python 3.10 or newer is required. Open PowerShell in this repository folder and run:

```powershell
py -3 -m pip install -r requirements.txt
.\CV7001_run_f1_windows.bat
```

Alternatively:

```powershell
py -3 .\CV7001_evaluate_f1.py --config .\CV7001_F1_study_config_PRIVATE.json --input-dir .\CV7001_F1_input --output-dir .\CV7001_F1_results --label-mode damage --no-sensitivity
```

For additional instructions, see [CV7001_F1_Updated_Instructions_EN.md](CV7001_F1_Updated_Instructions_EN.md).

## 4. Input images and classification colours

Each specimen uses **three** images: (i) an actual annotated specimen photograph with magenta damage outlines, (ii) CANUPO Only classification and (iii) R–C classification. In classification images, red denotes predicted damage and blue denotes predicted intact concrete.

Use the **original classification exports**. Do not evaluate recoloured, AI-enhanced, resized screenshots or presentation images unless their effects have been documented and accounted for.

## 5. Evaluation method

The F1 evaluation is an image-based **20 × 20 region-level comparison**. A 600 × 600 rectified specimen-face frame is used internally:

- Ground-truth positive: at least **8%** of the grid cell covered by the damage reference mask.
- Predicted positive: at least **15** red+blue classified pixels in a cell, of which at least **25%** are red.
- Magenta crack-line tolerance: **3%** of rectified specimen-face height.
- S02–S04: preliminary spalling-interior polygons, recorded in the JSON file.
- F1-score: `2*TP / (2*TP + FP + FN)`.

The F1-score measures agreement **at image-grid scale** between photographic reference damage regions and point-cloud classification renderings. It is not 3D per-point accuracy or a structural capacity assessment.

### Full-photo display versus evaluation alignment

The current Python version creates **two different quality-control outputs** for each specimen:

1. `S01_damage_review.png` — shows the **complete source photograph** in its original aspect ratio, padded (not cropped or enlarged to fill the view), together with ground-truth display, rectified CANUPO results and TP/FP/FN grids.
2. `S01_damage_alignment_check.png` — shows the source photograph with the specimen-face ROI, the **rectified scoring photograph**, and the **same** reference mask over the rectified photograph.
3. `S01_damage_GT_mask_rectified.png` — the **exact binary mask** used to create GT-positive grid cells for F1; white indicates damage and black indicates non-damage.

The same naming pattern is used for S02–S04. **Both GT overlays now come from exactly the same mask used by F1 scoring**, including the 3%-of-height line buffer and configured spalling-interior polygons. In `*_damage_review.png`, the mask is inverse-warped from the 600×600 scoring frame onto the **full original photograph**, without independently recreating polygons or buffer. In `*_damage_alignment_check.png`, the identical scoring mask is shown directly in the rectified scoring frame. Because these views use different perspectives, their *visible shapes* may still differ; this is expected and does not mean they use different labels. The raw mask PNG provides an additional audit trail. **Full-photo display does not modify the F1 calculation.** Scoring is based on the separately rectified ROI. The JSON records `photo_corners_xy` in the reference photo resolution (`photo_reference_size_wh`); the script automatically scales these coordinates if the *same photo* has been saved at another pixel resolution.

**Important:** Changing the photo crop, photograph viewpoint, CANUPO viewport or relative specimen position cannot be corrected just by resizing. Check every `*_alignment_check.png`. If the cyan/yellow ROI outline does not match the concrete face, update its corners in the JSON before interpreting the scores. A warning is printed when the input photo aspect ratio differs from the reference image by more than 5%.

## 6. Generated outputs

Running the batch launcher creates `CV7001_F1_results/` with:

| File | Purpose |
|:--|:--|
| `f1_comparison.csv` | CANUPO Only and R–C F1 (%) for each specimen, and change in percentage points |
| `per_model_metrics.csv` | TP, FP, FN, TN, precision, recall, F1, and photo ROI details |
| `S01_damage_review.png` ... `S04_damage_review.png` | Six-panel full-photo and classification checks |
| `S01_damage_alignment_check.png` ... `S04_damage_alignment_check.png` | ROI and the exact scoring GT in rectified reference coordinates |
| `S01_damage_GT_mask_rectified.png` ... `S04_damage_GT_mask_rectified.png` | Exact binary ground-truth masks used to derive grid labels |
| `S01_damage_grid_labels.npz` ... `S04_damage_grid_labels.npz` | Machine-readable 20×20 masks |
| `RUN_METADATA.json` | Configuration checksum and evaluation caveats |
| `parameter_sensitivity.csv` | Optional sensitivity analysis, when run without `--no-sensitivity` |

**The current CSV is the result for the actual supplied images.** A previous analysis used different image versions and produced a different F1-score table. Do not assume that an older report table matches the present images; compare your new `f1_comparison.csv` with the report before submission.

## 7. Reproducing the CloudCompare/CANUPO workflow

**Start with the saved CloudCompare projects.** `CV7001_CANUPO_Data Set/` already contains saved specimen projects with roughness analyses and manually labelled class samples, together with trained `.prm` classifiers. It is **not necessary to recompute Roughness or retrain a classifier simply to view the original study**.

For a detailed guide explaining how to open `S01.bin`, inspect stored `Roughness (r)` fields and `Crack`/`Conc` samples, examine saved `S01Canupo.bin` classification results, and optionally reload `S01_Canupo.prm` to classify a cloud, see **[CANUPO_Procedure.md](CANUPO_Procedure.md)**, especially Section 1.1. The instructions also apply to S02–S04 with the appropriate matching filenames.

CloudCompare v2.13.2 and its CANUPO plugin were used separately from Python. To reproduce the point-cloud part of this project, document and retain:

- Raw versus noise-cleaned scans (e.g. `S01OR.bin` is a noise-cleaned original point cloud).
- Class-labelled CANUPO training reference clouds and training settings.
- The four saved CANUPO `.prm` model files.
- Roughness radius, threshold selection, and which filtered point cloud was used for each R–C inference.
- The same starting input and evaluation basis for the CANUPO Only and R–C comparisons.
- How training and evaluation samples were separated.

The Python script does **not** itself retrain or run the CANUPO model. The repository must provide the appropriate CloudCompare GUI steps, model, and input data if the full training/inference workflow is to be reproduced.

## 8. Software and assistance disclosure

CloudCompare/CANUPO is third-party software. Python code and documentation for image-based F1 evaluation were developed with assistance from OpenAI ChatGPT. Reference damage outlines were marked manually, and the spalling-interior polygons are approximations for region-based evaluation. Scores depend on image quality, reference delineation and registration choices.
