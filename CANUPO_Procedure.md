# CloudCompare CANUPO Training and Classification Procedure

**Project:** Roughness-based CANUPO (SVM) for Concrete Surface Damage Detection  
**Course:** CV7001 Machine Learning Mini-Project  
**Software:** CloudCompare v2.13.2 (64-bit), CANUPO/qCANUPO plugin  
**Units:** metres (m) for point-cloud coordinates and neighbourhood radii  
**Scope:** S01–S04, one exposed concrete face per specimen

> **Reproducibility note.** This document records the workflow and values available from the project analysis, screenshots and saved-file names. Parameters that cannot yet be verified from the supplied records are marked **TO VERIFY** rather than assumed. The Python script supplied with this repository reproduces *image-based F1 evaluation*; CANUPO training and inference themselves are performed in CloudCompare.

## 1. Data and files

Four previously tested concrete specimens located outdoors at the NTU Construction Laboratory were captured using the LiDAR capability of an iPhone 16 Pro. S01 is primarily crack-dominated, while S02–S04 include cracking and spalling.

| Specimen | Analysed point-cloud size | Damage reference points | Intact-concrete reference points | Saved classifier |
|:--|--:|--:|--:|:--|
| S01 | 27,444 | 113 | 260 | `S01_Canupo.prm` |
| S02 | 37,990 | 52 | 829 | `S02_Canupo.prm` |
| S03 | 11,390 | 65 | 191 | `S03_Canupo.prm` |
| S04 | 9,984 | 54 | 188 | `S04_Canupo.prm` |

The reference-point counts above are the manually selected **damage / intact concrete training samples**, not counts of detected damage in the full cloud. These reference samples were selected from point clouds **before roughness-based filtering**.

The project folder includes the following types of files (the S01 names are examples):

| File | Role | Notes |
|:--|:--|:--|
| `S01.bin` | Saved original point-cloud project | Check the DB Tree to identify the source cloud(s). |
| `S01OR.bin` | Noise-removed original point cloud | **OR means noise-cleaned original**, not the roughness-filtered result. |
| `S01Canupo.bin` | Saved CANUPO project/result | Inspect the DB Tree to identify its classifier outputs. |
| `S01_Canupo.prm` | Trained CANUPO classifier | Retain with the corresponding specimen. |
| `S01.jpg` | Specimen photograph | The magenta-annotated evaluation image is stored separately in `CV7001_F1_input/`. |

Expected repository directories include `CV7001_CANUPO_Data Set/`, `CV7001_Original Scan Data set/` and `CV7001_F1_input/`. Preserve the existing `.bin` projects because they may contain the reference samples, computed scalar fields, filtered clouds and classification outputs.

**TO VERIFY before submission:** Record the exact cloud entity used as the common starting point for CANUPO Only and R–C, and the noise-removal command/parameters applied to each source scan. The file suffix `OR` alone does not establish which entity was used in every run.

## 1.1. Quick start: using the saved `CV7001_CANUPO_Data Set/` files (recommended)

**You do not need to repeat the Roughness sweep, manually select the training samples, or retrain the CANUPO classifiers just to inspect the saved study.** The project's existing CloudCompare `.bin` files and `.prm` classifiers are the primary reproducibility materials. This quick-start explains how a reviewer can open the existing results first, and then optionally rerun classification or preprocessing.

### A. Locate the existing files

From the root of the repository, open the folder `CV7001_CANUPO_Data Set/`. The saved file groups follow the naming convention below (examples from S01; corresponding S02–S04 files follow the same pattern where present):

| File | Open with / how to use | Expected content |
|:--|:--|:--|
| `S01.bin` | **CloudCompare > File > Open** | Saved specimen project. Inspect the DB Tree for the analysed cloud, LNR-specific roughness clouds/fields, and manually selected `Crack` / `Conc` reference samples. |
| `S01OR.bin` | **CloudCompare > File > Open** | Noise-removed original scan; **not** the roughness-filtered cloud. |
| `S01Canupo.bin` | **CloudCompare > File > Open** | Saved CANUPO analysis/classification project. Confirm the classification entities and scalar field labels in the DB Tree. |
| `S01_Canupo.prm` | Load from **CANUPO > Classify** (not as a point cloud) | Already trained S01 classifier, ready to reuse for inference. |
| `S01.jpg` | Any image viewer | Physical specimen photograph; distinguished from the marked GT and class-rendering images in `CV7001_F1_input/`. |

The exact internal entity names may differ among specimens. Use **what actually appears in the CloudCompare DB Tree**, rather than assuming that every `.bin` contains identical sub-objects. Opening a `.bin` project does **not** automatically run classification, and a `.prm` is a classifier, **not** a point-cloud project.

### B. Open and inspect the saved point-cloud, Roughness and class samples

1. Launch **CloudCompare v2.13.2** and open `CV7001_CANUPO_Data Set/S01.bin` via **File > Open**. Repeat for `S02.bin`, `S03.bin` and `S04.bin` as required (prefer one specimen at a time to prevent DB Tree confusion).
2. Expand the `.bin` entry in the **DB Tree**. Look for the specimen cloud (`S01`), roughness-radius variants (for example, `S01_0.010`, `S01_0.015`, `S01_0.020` and the selected LNR), and the manually labelled training clouds `Crack` and `Conc` where saved.
3. Use the **visibility checkboxes** to show one entity at a time and select it to inspect **Properties > Points**. Do not add or merge point counts from overlapping visible clouds.
4. To inspect an existing Roughness calculation, select the corresponding LNR cloud, find **Properties > Scalar Fields > Active**, and choose the saved `Roughness (radius)` scalar field if present. Enable the scalar-field colour scale and open the histogram from CloudCompare's scalar-field histogram control. **No new Roughness computation is necessary when the saved field exists.**
5. For the 50%-cut output, inspect the saved **extracted/filtered cloud** for that radius, if present (e.g., an entity with `extract` in its name). Compare its point count and spatial pattern with the unfiltered candidate and the values in `Appendix_Data Analysis.xlsx`. An extracted cloud may retain a Roughness scalar field, but do not assume it does.
6. Select the original `Crack` and `Conc` reference sample entities separately. Verify their label role and point counts against the training-reference table in Section 1. These samples were made **before** the roughness cut; they are not newly selected from each 50%-filtered cloud.
7. Repeat the inspection for the **selected LNR** of each specimen:

| Specimen | Selected LNR | What to inspect in the saved project |
|:--|--:|:--|
| S01 | **0.025 m** | `Roughness (0.025)` and its extracted candidate, if saved |
| S02 | **0.010 m** | `Roughness (0.010)` and its extracted candidate, if saved |
| S03 | **0.015501 m** | `Roughness (0.015501)` and its extracted candidate, if saved |
| S04 | **0.030 m** | `Roughness (0.030)` and its extracted candidate, if saved |

**Important:** A roughness-radius variant and a filtered/extracted cloud are **not necessarily the same entity**. The first stores computed features; the second contains the points remaining after the cutoff. If the expected extracted entity is absent, use Section 6 to recreate it from the saved Roughness scalar field and the recorded numerical threshold (if that threshold is known).

### C. Inspect existing CANUPO results without retraining

1. Open `CV7001_CANUPO_Data Set/S01Canupo.bin` in CloudCompare.
2. Expand its DB Tree and identify the saved classification result cloud(s). Select the relevant entity and inspect the available **scalar fields/class labels**; consult the saved display colours or class legend to identify **damage** and **intact concrete**.
3. Display the baseline and R–C outputs separately, if both are included. If the `.bin` only stores one of the methods, refer to the saved images `CV7001_F1_input/S01_CANUPO.*` and `CV7001_F1_input/S01_RC.*` for the documented visual comparison. Do not infer that one file contains two different classifiers without checking the DB Tree.
4. Compare the saved point-cloud classification visually with the corresponding specimen photograph. For numerical, image-based agreement, run the Python evaluator described in Section 8.

**This is the fastest route for a reviewer:** the saved project allows inspection of the work already performed. Retraining the SVM is **optional** if the purpose is to inspect saved classifications; exact retraining settings are still needed for a complete rerun from raw data.

### D. Reuse the saved `.prm` model to classify a point cloud

1. Open the specimen's saved source/cloud project (typically the agreed noise-cleaned or otherwise documented starting point cloud).
2. Select the target cloud to classify in the DB Tree. For the baseline, select the **non-roughness-filtered starting cloud**; for R–C, select the **saved cloud retained after the selected LNR's roughness cutoff**.
3. Open the **CANUPO Classify** dialog from the plugin toolbar/menu, and select the corresponding saved `S01_Canupo.prm` file. Match specimen IDs; do not accidentally load another specimen's model.
4. Confirm the classification direction/label legend and the study's classification options, then execute classification. The two class colours in the report/export are **red for damage** and **blue for intact concrete**.
5. Save the resulting project to a **new filename** (for example, `S01_rerun.bin`) so that the original `S01.bin`, `S01OR.bin`, `S01Canupo.bin` and `.prm` files remain unmodified.
6. Repeat for S02–S04 as needed. If the baseline and R–C originally used different trained models, use the model documented for each branch and record the difference rather than silently substituting one classifier.

**No `.prm` model retraining is required for classification with the saved model.** Training again is only needed if the goal is to reconstruct training from the original labelled point clouds (see Sections 3–4). The `.prm` file does not automatically apply Roughness filtering; these are separate stages.

### E. Reproduce the figure-level F1 results from the existing exports

The Python evaluation **does not need to read `.bin` or `.prm`**. It reads three exported/annotated images for each specimen from `CV7001_F1_input/`:

```text
S01.jpg           S01_CANUPO.jpg           S01_RC.jpg
S02.jpg           S02_CANUPO.jpg           S02_RC.jpg
S03.jpg           S03_CANUPO.jpg           S03_RC.jpg
S04.jpg           S04_CANUPO.jpg           S04_RC.jpg
```

These extensions are examples; the script also recognises matching PNG and other supported image formats. Run from the repository's root directory:

```powershell
py -3 -m pip install -r requirements.txt
.\CV7001_run_f1_windows.bat
```

Then open `CV7001_F1_results/f1_comparison.csv`, `*_damage_review.png`, `*_damage_alignment_check.png`, and `*_damage_GT_mask_rectified.png`. This is a separate image-grid F1 analysis of the **saved classification renderings**, not a new CANUPO training run.

### F. Troubleshooting and preservation

- **No clouds appear:** verify that the correct `.bin` was opened, its DB Tree visibility checkboxes are enabled, and **Zoom to fit** is used. Close unrelated specimens to avoid overlapping views.
- **No `Roughness (r)` field appears:** select the *radius-specific source cloud*, not an unrelated sample or extracted entity; inspect its list of scalar fields. If no field is saved, Section 6 explains how to compute it again.
- **The `.prm` file does not open as a point cloud:** this is expected; load it through the CANUPO **Classify** dialog.
- **Classification colours/labels seem reversed:** check the saved model's class label mapping; do not swap colours solely to improve visual appearance or F1-score.
- **Baseline and R–C look identical:** confirm that the correct **filtered point cloud** was selected for R–C rather than just a cloud with the Roughness scalar field displayed.
- **Python F1 results differ from the report:** verify that `CV7001_F1_input/` contains the same image versions and that the JSON crop/registration settings are unchanged. Use the generated alignment/GT checks.
- **Keep originals unchanged:** saved `.bin`, `.prm`, selected training clouds and source photographs should remain available as the evidence of the original GUI analysis.

## 2. Prepare the source cloud in CloudCompare

1. Open **CloudCompare v2.13.2** and load the relevant `.bin` project through **File > Open**.
2. In the DB Tree, identify the specimen's analysed face, the raw scan and (where available) the noise-cleaned original cloud.
3. Hide unrelated objects. Confirm that coordinates are in **metres**, the face orientation is correct and the analysed point count matches the project record.
4. Preserve the raw and noise-cleaned clouds. Work on a **clone** for each comparison so that changes to the R–C branch do not alter the baseline.
5. Record the chosen input entity name and whether a common noise-removal stage was applied to both branches.

**Input equivalence:** The comparison should use the same starting geometry and cleaning stage, with roughness filtering being the intended difference between methods. If the recorded study used different source clouds, report this explicitly instead of silently treating them as identical.

## 3. Create the manually labelled CANUPO training samples

1. On the original, **unfiltered** point cloud, use CloudCompare's segmentation/scissors tool to select representative damage locations (cracks and, where relevant, spalling) and representative intact concrete.
2. Keep the selected points as separate entities, for example `Crack` (or `Damage`) and `Conc`.
3. Where multiple areas belong to one class, merge those extracted subsets within that class without merging the two classes together.
4. Verify the two class labels and sample counts against the table in Section 1.
5. Save a copy of the project containing both class-reference clouds.

**Important:** In this project, the damage and intact-concrete *training reference clouds* were **not re-extracted after the 50% roughness cutoff**. Do not describe them as training samples generated from the roughness-filtered output.

The reference labels are selected samples for supervised classifier training. They are distinct from the annotated photographs and image-grid reference masks used later for F1-score evaluation.

## 4. Train and save the CANUPO (SVM) classifier

1. In CloudCompare, open **Plugins > CANUPO/qCANUPO > Train classifier**. (The displayed plugin label may differ slightly between installations.)
2. In the **Data** section, assign `Conc` to the intact-concrete class and `Crack`/`Damage` to the damage class. Confirm the **class-number mapping** in the resulting legend before saving.
3. Specify CANUPO's multi-scale **descriptor scales** (ramp or explicit list). These are **separate from the roughness LNR** used in Section 6.
4. In **Advanced**, select **Dimensionality** as the classification parameter. The recorded training dialog shows **Max core points = 10,000**.
5. Leave **Use original cloud for descriptors** **unchecked**, as in the user's successful training configuration. This setting should not be confused with the origin of the manually selected training references.
6. Run training. Inspect the plotted class groups and classification boundary. Review the Statistics output if it is available, and record any manual boundary edits or scale removals.
7. Click **Save** and retain the classifier as `S01_Canupo.prm`, `S02_Canupo.prm`, `S03_Canupo.prm` or `S04_Canupo.prm`, respectively. Do not overwrite these with classifiers from another specimen.

**TO VERIFY for exact retraining:** The final descriptor **Min / Step / Max** or explicit scale list, whether any scales were removed in the result window, the final class-number mapping, and whether the classification boundary was manually edited. Earlier training screenshots are not sufficient to establish that every specimen used identical final scales. Record these from the saved training sessions or classifier files instead of substituting arbitrary values.

> **Technical distinction:** The local neighbourhood radius \(r\) defines the search radius for CloudCompare **roughness calculation**. It is **not an SVM kernel hyperparameter**. Roughness filtering changes the point cloud presented to CANUPO and can therefore affect its classification.

## 5. Run the baseline: CANUPO Only

1. In the DB Tree, select the agreed **unfiltered-with-respect-to-roughness** starting cloud (see Section 2). It may be noise-cleaned if noise cleaning was a common preprocessing step.
2. Open **Plugins > CANUPO/qCANUPO > Classify**.
3. Load the corresponding specimen's `.prm` classifier.
4. Choose the cloud/core-point option used in the study. For the fully reproduced study, retain the same classification option and settings for both workflows, except where the original records demonstrate otherwise.
5. Run classification; confirm which class ID corresponds to damage and which to intact concrete.
6. Save the resulting classified cloud and export an **unaltered classification rendering** with **red = predicted damage** and **blue = predicted intact concrete**. Record the viewing direction, framing and point size.

The images named `S01_CANUPO.jpg` (similarly S02–S04) in `CV7001_F1_input/` represent the baseline's exported classification view. The file extensions may be JPG or PNG.

**TO VERIFY:** Identify exactly which `.prm` was used for the baseline and whether that same model was applied in the R–C branch. If the models were trained separately, record that fact: the comparison then reflects both preprocessing and classifier-training differences.

## 6. Run roughness analysis and choose the LNR

1. Clone the same agreed source cloud for each tested LNR; do not repeatedly filter the previously filtered output.
2. Select the clone and open **Tools > Other > Compute geometric features**.
3. Enable **Roughness** and enter the candidate **Local neighbourhood radius (LNR)** in metres. A **Number of neighbours** feature was also enabled in some recorded dialogs; it is supplementary and should not be conflated with the roughness field.
4. Run the computation. Confirm that the active scalar field is `Roughness (r)` and inspect its histogram.
5. Choose an approximately **50% cutoff in the roughness-value distribution**, as used in the project, and record the **numerical roughness scalar threshold** for this LNR.
6. Using **Edit > Scalar fields > Filter by value**, extract the higher-roughness portion and save it as a separate cloud. This removes low-roughness points from the R–C candidate.
7. Inspect the retained damage-region visibility and record the number of retained crack/damage reference points and intact-concrete reference points. Repeat for all tested candidate radii.
8. Select the LNR mainly by how clearly the retained damage features can be identified across the evaluated candidates. Track total removal separately rather than using it alone to optimise the choice.

**Threshold interpretation:** The approximately 50% histogram cutoff is a **threshold-selection convention**, **not a guarantee** that exactly 50% of the complete PCD is removed. The final PCD removal ratio may differ because of the point selection, filtering range and any additional processing. Record the histogram threshold and final counts separately.

The selected study conditions are:

| Specimen | Selected LNR (m) | Selected LNR (mm) | Recorded PCD removed (%) | Damage/crack retained (%) | Intact-concrete retained (%) |
|:--|--:|--:|--:|--:|--:|
| S01 | 0.025 | 25.0 | 48.48 | 6.19 | 10.77 |
| S02 | 0.010 | 10.0 | 67.75 | 21.15 | 23.76 |
| S03 | 0.015501 | 15.501 | 46.93 | 78.46 | 12.04 |
| S04 | 0.030 | 30.0 | 51.04 | 51.85 | 4.79 |

Definitions used in the data analysis:

- **Class retention (%)** = retained class-reference points / original class-reference points × 100.
- **Crack visibility contrast (percentage points)** = crack/damage retention (%) − intact-concrete retention (%).
- **PCD removed (%)** = reduction in total point count relative to the stated source cloud; this is recorded but is **not** the sole LNR selection score.

The complete candidate sweeps and recorded histograms/results are retained in `Appendix_Data Analysis.xlsx` and/or in the saved CloudCompare project files.

**TO VERIFY:** Record the numerical scalar threshold and exact retained-cloud entity for each selected LNR, especially where histogram cutoff percentages and net PCD removal differ. The table above records study outputs but does not by itself allow the filtering operation to be reconstructed exactly.

## 7. Run Roughness–CANUPO (R–C) classification

1. Select the cloud retained after roughness filtering with the specimen's selected LNR.
2. Run **Plugins > CANUPO/qCANUPO > Classify**.
3. Load the classifier recorded for the R–C run. Ideally, use the **same trained `.prm`** as the baseline to isolate the effect of roughness filtering; however, verify this against the actual saved project before claiming that this was done.
4. Use the recorded class/core-point settings, verify the damage-class label and run classification.
5. Save the classified result. Export the **original unenhanced rendering** using the same view/framing and point-size convention as the baseline as far as practicable.
6. Keep the R–C image as `S01_RC.jpg` (similarly S02–S04) in `CV7001_F1_input/`.

Do **not** use artificially recoloured or generated presentation images as inputs to the quantitative F1-score script.

## 8. Image-based validation (Python; separate from CANUPO)

The GitHub Python code **does not train or classify point clouds**. It compares the previously exported CANUPO Only and R–C renderings with damage outlines manually annotated on specimen photographs.

Each specimen contributes three image inputs, with these example filenames (actual supported extension may vary):

```text
CV7001_F1_input/
  S01.jpg          S01_CANUPO.jpg          S01_RC.jpg
  S02.jpg          S02_CANUPO.jpg          S02_RC.jpg
  S03.jpg          S03_CANUPO.jpg          S03_RC.jpg
  S04.jpg          S04_CANUPO.jpg          S04_RC.jpg
```

Evaluation settings in `CV7001_F1_study_config_PRIVATE.json`:

| Setting | Value |
|:--|:--|
| Rectified specimen-face frame | 600 × 600 pixels |
| Evaluation grid | 20 × 20 cells |
| Ground-truth positive-cell threshold | ≥ 8% of cell area covered by reference damage mask |
| Magenta crack-line buffer | 3% of rectified face height |
| Predicted damage-cell threshold | ≥ 15 coloured (red + blue) pixels, with ≥ 25% red |
| Damage reference | S01: annotated cracking; S02–S04: annotated cracks plus provisional spalling-interior polygons |

Run from PowerShell in the repository root:

```powershell
py -3 -m pip install -r requirements.txt
.\CV7001_run_f1_windows.bat
```

The script creates `CV7001_F1_results/`, including `f1_comparison.csv`, `per_model_metrics.csv`, per-specimen `*_damage_review.png`, `*_damage_alignment_check.png`, and `*_damage_GT_mask_rectified.png`. The last two are important visual checks: both use the same ground-truth mask, although one displays it in the rectified scoring frame and the other review figure also contains the full-photo perspective.

The latest F1-scores **reported from the user's Windows run with the corresponding local images** are:

| Specimen | CANUPO Only F1 (%) | R–C F1 (%) | Change (percentage points) |
|:--|--:|--:|--:|
| S01 | 41.06 | 38.25 | −2.81 |
| S02 | 56.58 | 75.70 | +19.12 |
| S03 | 72.46 | 88.89 | +16.43 |
| S04 | 54.76 | 71.63 | +16.87 |

These are **image-grid damage-region agreement scores** and depend on registration, annotation boundaries and input rendering. They are **not** 3D pointwise classification accuracy or a direct estimate of structural capacity. Report the numbers generated by the actual version of the input files and JSON in the repository, not older provisional values.

## 9. Submission audit: items to verify

The study's major processing stages, saved classifier files and F1 evaluator are available, but exact CANUPO/CloudCompare reruns require a few records that are not established by the screenshots alone:

- [ ] Identify the exact baseline and R–C starting cloud entity for each specimen, including any common noise cleaning.
- [ ] Record the noise-removal tool/parameters, or clearly mark the cleaned PCD as a provided intermediate.
- [ ] Confirm the saved final CANUPO descriptor scales, class label mapping, core-point option and any boundary edits.
- [ ] Confirm which `.prm` file was applied to CANUPO Only and R–C.
- [ ] Record the selected roughness scalar cutoff value and retained cloud for each LNR.
- [ ] Verify that uploaded `.bin`, `.prm` and input images can be opened, and that the F1 code runs from the uploaded repository.
- [ ] Check all four full-photo/rectified alignment and exact-GT-mask review images.
- [ ] Keep the final report's F1 table in sync with the repository's `f1_comparison.csv`.

If a parameter cannot be recovered, state that it is unrecorded and provide the saved intermediate `.bin` and `.prm` files rather than inventing a numeric value.

## 10. External software documentation and disclosure

The analysis uses CloudCompare and the CANUPO plugin; no standalone implementation of CANUPO's SVM is claimed. The F1 evaluation script and project documentation were prepared with AI assistance and require researcher review of reference-mask and image-alignment decisions.

Official CloudCompare references:

- [CANUPO plugin: training, saving `.prm` and classification](https://cloudcompare.org/doc/wiki/index.php/CANUPO_%28plugin%29)
- [Compute geometric features: Roughness and LNR](https://www.cloudcompare.org/doc/wiki/index.php/Compute_geometric_features)
- [Roughness definition](https://cloudcompare.org/doc/wiki/index.php/Roughness)
- [Scalar fields: Filter by value](https://cloudcompare.org/doc/wiki/index.php/Scalar_fields%5CFilter_by_Value)
