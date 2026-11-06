#!/usr/bin/env python3
"""Reproducible *provisional* 2-D region-based damage F1 evaluation.

Two workflows are compared against the SAME annotated photograph:
  - only: CANUPO classifier without roughness preprocessing
  - rc:   roughness-conditioned CANUPO visualisation

These are image-rendering, photo-registration and manual-annotation-dependent
metrics, NOT point-cloud accuracy or independent external validation.

The six-panel review shows the FULL source photograph without cropping. Its GT
panel projects the exact scoring mask from the 600x600 rectified specimen face
back to the full photo; the alignment-check overlay uses that same GT mask. Stored photo
corners are scaled to the actual image resolution using the reference dimensions
recorded in the JSON configuration. Changes of viewpoint/crop require review.

Usage:
  python CV7001_evaluate_f1.py
  # optional: python CV7001_evaluate_f1.py --label-mode both

The program recognises these actual Windows Explorer names without renaming:
  S01 / S01_CANUPO / S01_RC (and S02, S03, S04),
with .jpg, .jpeg, .png, .bmp, .tif, .tiff or .webp extensions.

Code: stdlib + numpy, OpenCV. Does not optimise thresholds to favour either model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn) if 2 * self.tp + self.fp + self.fn else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"TP":self.tp,"FP":self.fp,"FN":self.fn,"TN":self.tn,
                "precision":round(self.precision, 6),"recall":round(self.recall, 6),
                "f1":round(self.f1, 6),"f1_percent":round(self.f1*100, 2)}


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_input_image(base: Path, spec: dict, role: str) -> Path:
    """Resolve original config names OR current user file names, regardless of extension.

    Accepted aliases for each specimen ID (case-insensitive):
      GT:    S01_gt.* OR S01.*
      Only:  S01_only.* OR S01_CANUPO.*
      R-C:   S01_rc.*

    The configured exact filename takes precedence. Ambiguous duplicate stems are
    rejected instead of silently choosing a potentially wrong input file.
    """
    configured = Path(spec[role])
    if configured.is_absolute():
        if configured.is_file():
            return configured
        raise FileNotFoundError(f"Configured absolute input does not exist: {configured}")

    direct = base / configured
    if direct.is_file():
        return direct
    if not base.is_dir():
        raise NotADirectoryError(f"Input directory was not found: {base}")

    sid = str(spec["id"])
    aliases = {
        "ground_truth_photo": [f"{sid}_gt", sid],
        "canupo_only": [f"{sid}_only", f"{sid}_CANUPO"],
        "roughness_canupo": [f"{sid}_rc", f"{sid}_RC"],
    }
    if role not in aliases:
        raise ValueError(f"Unknown image role: {role}")

    # First accept the configured stem with a different image extension, then
    # fall back to the Windows Explorer names supplied by the user.
    requested_stem = configured.stem if configured.suffix.lower() in IMAGE_SUFFIXES else configured.name
    stems = list(dict.fromkeys([requested_stem.lower(), *(a.lower() for a in aliases[role])]))
    images = sorted(
        (p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda x: x.name.lower(),
    )
    for stem in stems:
        matches = [p for p in images if p.stem.casefold() == stem.casefold()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous images for {sid} {role}: "
                + ", ".join(p.name for p in matches)
                + ". Remove duplicates or give an exact filename in the config."
            )
    raise FileNotFoundError(
        f"Could not find an image for {sid} / {role} in {base}. "
        f"Tried stems: {', '.join(stems)}. "
        f"Available images: {', '.join(p.name for p in images) or '(none)'}"
    )


def read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input image: {path}")
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Unable to decode image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed saving image to {path}")


def mask_magenta(img: np.ndarray) -> np.ndarray:
    """Identify conspicuous user-drawn violet/magenta markup, not grey concrete.

    RGB threshold is intentionally fixed across all specimens; verify diagnostic
    overlays before interpreting results. Changes must be preregistered.
    """
    a = img.astype(np.int16)
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    return (r >= 95) & (b >= 68) & (r >= g + 48) & (b >= g + 36) & (r >= b * 0.63)


def masks_red_blue(img: np.ndarray, brightness_min: int) -> tuple[np.ndarray,np.ndarray]:
    """Binary red/blue dominance masks, common pixel threshold for both methods."""
    a = img.astype(np.int16)
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    red = (r >= brightness_min) & (r >= b + 8) & (r >= g + 8)
    blue = (b >= brightness_min) & (b >= r + 8) & (b >= g + 8)
    return red,blue


def bbox_from_colours(img: np.ndarray, brightness_min: int) -> list[list[int]]:
    red,blue=masks_red_blue(img,brightness_min)
    yy,xx=np.where(red|blue)
    if not len(xx):
        raise ValueError("No red/blue pixels detected in prediction rendering")
    # Use detected extrema, which users can replace by explicit corners in the config.
    x0,x1=int(xx.min()),int(xx.max())
    y0,y1=int(yy.min()),int(yy.max())
    if (x1-x0)<100 or (y1-y0)<100:
        raise ValueError("Detected prediction ROI is unexpectedly small; verify input")
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def warp_square(img: np.ndarray, corners: list[list[float]], size: int,
                interpolation: int=cv2.INTER_NEAREST) -> np.ndarray:
    """Corners TL,TR,BR,BL in the ORIGINAL input image pixel coordinates."""
    src=np.array(corners,dtype=np.float32)
    if src.shape!=(4,2):
        raise ValueError("corners must be a TL,TR,BR,BL array of four [x,y] points")
    dst=np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]],dtype=np.float32)
    H=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,H,(size,size),flags=interpolation,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0)



def resolve_photo_corners(spec: dict, photo: np.ndarray) -> tuple[list[list[float]], float]:
    """Scale the recorded photo ROI to the current image's pixel resolution.

    This corrects the common issue of using pixel coordinates drawn on a
    resized 819x1006 annotation with a larger copy of the same photograph.
    It does not fix a change in crop, camera perspective or photograph content.

    Returns the scaled corners and percentage aspect-ratio disagreement.
    """
    h, w = photo.shape[:2]
    ref = spec.get("photo_reference_size_wh")
    src_corners = np.asarray(spec["photo_corners_xy"], dtype=np.float64)
    if src_corners.shape != (4, 2):
        raise ValueError("photo_corners_xy must specify four [x, y] corners")
    if not ref:
        return src_corners.tolist(), 0.0
    ref_w, ref_h = map(float, ref)
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError("photo_reference_size_wh dimensions must be positive")
    aspect_error = 100.0 * abs(w/h - ref_w/ref_h) / (ref_w/ref_h)
    corners = src_corners * np.array([w/ref_w, h/ref_h])
    if np.min(corners[:, 0]) < -1 or np.min(corners[:, 1]) < -1 or np.max(corners[:, 0]) > w + 1 or np.max(corners[:, 1]) > h + 1:
        raise ValueError(f"ROI corners outside {w}x{h} input photo. Check photo_reference_size_wh")
    return corners.tolist(), aspect_error


def fit_image_to_square(img: np.ndarray, size: int, bg: tuple[int, int, int]=(245,245,245)) -> np.ndarray:
    """Show the full original photo without cropping, padded to a square canvas."""
    h, w = img.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def fit_mask_to_square(mask: np.ndarray, size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(mask.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    canvas = np.zeros((size, size), dtype=bool)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def project_scoring_gt_to_photo(gt_rect: np.ndarray, photo: np.ndarray,
                                corners: list[list[float]]) -> np.ndarray:
    """Project the EXACT rectified GT mask used for F1 back to the full photo.

    The source is the *same* scoring mask that is used for the grid labels and
    alignment-check image (including crack-line tolerance and spalling polygons).
    Perspective re-projection may introduce a few raster-level edge differences.
    """
    size = gt_rect.shape[0]
    if gt_rect.ndim != 2 or gt_rect.shape != (size, size):
        raise ValueError('Expected a square rectified GT mask')
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=np.float32)
    photo_to_rect = cv2.getPerspectiveTransform(src, dst)
    rect_to_photo = np.linalg.inv(photo_to_rect)
    mask_orig = cv2.warpPerspective(gt_rect.astype(np.uint8) * 255,
                                    rect_to_photo,
                                    (photo.shape[1], photo.shape[0]),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    return mask_orig > 0


def overlay_scoring_gt(photo_or_roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Uniform QC colouring for the SAME scoring GT mask in both visualisations.

    Magenta fill denotes GT pixels; yellow outlines are *display-only* edges.
    The F1 score is calculated from the original mask, not from the overlay.
    """
    if mask.shape != photo_or_roi.shape[:2]:
        raise ValueError('Ground-truth mask must match image dimensions')
    view = photo_or_roi.copy()
    if np.any(mask):
        tint = np.array([245., 45., 185.], dtype=np.float32)
        view[mask] = np.rint(0.58 * view[mask].astype(np.float32) +
                              0.42 * tint).astype(np.uint8)
        edge = cv2.morphologyEx((mask.astype(np.uint8) * 255),
                                cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        view[edge] = [255, 215, 55]
    return view


def make_preview_panels(photo: np.ndarray, gt_rect: np.ndarray,
                        corners: list[list[float]], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Show full original photo plus the actual scoring GT projected onto it.

    No independent spalling polygon drawing or independently buffered magenta
    annotation is used for the full-photo GT panel. Both QC outputs therefore
    derive from one, identical ground-truth mask, produced by make_gt_mask().
    """
    marked_preview = fit_image_to_square(photo, size)
    gt_original = project_scoring_gt_to_photo(gt_rect, photo, corners)
    gt_preview = overlay_scoring_gt(fit_image_to_square(photo, size),
                                    fit_mask_to_square(gt_original, size))
    return marked_preview, gt_preview


def make_gt_mask(photo: np.ndarray, spec: dict, size: int,
                 buffer_fraction: float, damage_mode: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Make GT from user strokes; optionally add *provisional* hand-drawn spall polygons.

    For all photos the user supplied magenta outlines. Polygon interiors are manually
    interpreted, provisional and must be reviewed; strokes remain original user labels.
    """
    magenta_source=mask_magenta(photo)
    photo_rect=warp_square(photo,spec["photo_corners_xy"],size,cv2.INTER_LINEAR)
    line_rect=warp_square(magenta_source.astype(np.uint8)*255,
                          spec["photo_corners_xy"],size,cv2.INTER_NEAREST)>0
    # Dilation gives spatial tolerance in the SAME rectified coordinates for both models.
    radius=max(0,round(buffer_fraction*size))
    if radius:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        line_rect=cv2.dilate(line_rect.astype(np.uint8),kernel)>0
    poly_mask=np.zeros((size,size),dtype=np.uint8)
    if damage_mode=="damage":
        for polygon in spec.get("provisional_spall_polygons_norm",[]):
            points=np.array([[round(float(x)*(size-1)),round(float(y)*(size-1))]
                             for x,y in polygon], dtype=np.int32)
            if len(points)<3:
                raise ValueError("spall polygon must have at least 3 vertices")
            if (points<0).any() or (points>=size).any():
                raise ValueError("spall polygon normalized vertices must be within [0,1]")
            cv2.fillPoly(poly_mask,[points],255)
    combined=line_rect | (poly_mask>0)
    return photo_rect, combined, poly_mask>0


def warp_predictions(img: np.ndarray, corners: list[list[float]], size: int,
                     brightness_min:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    red,blue=masks_red_blue(img,brightness_min)
    r=warp_square(red.astype(np.uint8)*255,corners,size)>0
    b=warp_square(blue.astype(np.uint8)*255,corners,size)>0
    # Colour renderings may have black gaps. They are not assumed to be class 0
    # at pixel scale, but cells with no classified pixels are predicted negative.
    return r,b,(r|b)


def binary_cell_labels(gt_mask: np.ndarray, red_mask: np.ndarray,
                       blue_mask:np.ndarray, grid_n:int, gt_fraction:float,
                       red_fraction:float, min_classified_pixels:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Score every cell, including cells with zero R-C points (predict negative).

    The denominator for a predicted positive is (red+blue) in each cell; class
    share is not a true point fraction because this is a rendered screenshot.
    """
    side=gt_mask.shape[0]
    if gt_mask.shape!=(side,side) or side%grid_n!=0:
        raise ValueError("square mask size must be divisible by grid_size")
    if red_mask.shape!=gt_mask.shape or blue_mask.shape!=gt_mask.shape:
        raise ValueError("prediction masks and GT must be the same size")
    gt=np.zeros((grid_n,grid_n),dtype=bool)
    predicted=np.zeros_like(gt)
    red_cell_share=np.zeros_like(gt,dtype=np.float32)
    step=side//grid_n
    for iy in range(grid_n):
        for ix in range(grid_n):
            sl=np.s_[iy*step:(iy+1)*step,ix*step:(ix+1)*step]
            gt[iy,ix] = (np.count_nonzero(gt_mask[sl])/float(step*step))>=gt_fraction
            n_red=int(np.count_nonzero(red_mask[sl]))
            n_blue=int(np.count_nonzero(blue_mask[sl]))
            denom=n_red+n_blue
            red_cell_share[iy,ix]=n_red/denom if denom else 0.0
            predicted[iy,ix]=(denom>=min_classified_pixels and red_cell_share[iy,ix]>=red_fraction)
    return gt,predicted,red_cell_share


def compute_counts(gt:np.ndarray,pred:np.ndarray)->Counts:
    if gt.shape!=pred.shape:raise ValueError("gt and prediction shapes differ")
    return Counts(int(np.count_nonzero(gt & pred)),int(np.count_nonzero(~gt & pred)),
                  int(np.count_nonzero(gt & ~pred)),int(np.count_nonzero(~gt & ~pred)))


def render_diagnostic(marked_preview: np.ndarray, gt_preview: np.ndarray,
                      only_red: np.ndarray, only_blue: np.ndarray, only_pred: np.ndarray,
                      rc_red: np.ndarray, rc_blue: np.ndarray, rc_pred: np.ndarray,
                      gt_grid: np.ndarray, sid: str, modes: str) -> np.ndarray:
    """Six panels; full-image photos fit without cropping, resizing with padding."""
    size = marked_preview.shape[0]
    title_h, border = 40, 8

    def with_title(img: np.ndarray, label: str) -> np.ndarray:
        panel = np.zeros((size + title_h, size, 3), dtype=np.uint8)
        panel[:title_h] = [20, 24, 30]
        panel[title_h:] = img
        cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    .61, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    def pred_vis(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        vis = np.zeros((size, size, 3), dtype=np.uint8)
        vis[blue] = [45, 82, 230]
        vis[red] = [240, 48, 55]
        return vis

    def cm_vis(pred: np.ndarray) -> np.ndarray:
        cell_h = size // gt_grid.shape[0]
        grid_img = np.full((size, size, 3), [25, 28, 33], dtype=np.uint8)
        for iy in range(gt_grid.shape[0]):
            for ix in range(gt_grid.shape[1]):
                if gt_grid[iy, ix] and pred[iy, ix]:
                    color = (50, 174, 115)  # true positive
                elif not gt_grid[iy, ix] and pred[iy, ix]:
                    color = (235, 159, 54)  # false positive
                elif gt_grid[iy, ix] and not pred[iy, ix]:
                    color = (177, 88, 199)  # false negative
                else:
                    color = (56, 62, 71)  # true negative
                y0, x0 = iy * cell_h, ix * cell_h
                grid_img[y0:y0+cell_h, x0:x0+cell_h] = color
        for t in range(0, size + 1, cell_h):
            cv2.line(grid_img, (min(t,size-1), 0), (min(t,size-1),size-1), (15,15,15), 1)
            cv2.line(grid_img, (0,min(t,size-1)), (size-1,min(t,size-1)), (15,15,15), 1)
        return grid_img

    panels = [
        with_title(marked_preview, f'{sid}  Marked photo (FULL view)'),
        with_title(gt_preview, f'{sid}  F1 scoring GT (projected FULL view)'),
        with_title(pred_vis(only_red, only_blue), 'CANUPO Only (red = damage)'),
        with_title(pred_vis(rc_red, rc_blue), 'R-C (red = damage)'),
        with_title(cm_vis(only_pred), 'Only: TP green / FP orange / FN purple'),
        with_title(cm_vis(rc_pred), 'R-C: TP green / FP orange / FN purple'),
    ]
    panel_h = size + title_h
    board = np.full((panel_h * 3 + border * 4,
                     size * 2 + border * 3, 3), 245, dtype=np.uint8)
    for i, panel in enumerate(panels):
        row, col = divmod(i, 2)
        y, x = border + (panel_h + border)*row, border + (size + border)*col
        board[y:y+panel_h, x:x+size] = panel
    return board


def render_alignment_check(photo: np.ndarray, photo_rect: np.ndarray,
                           gt_rect: np.ndarray, corners: list[list[float]],
                           size: int, sid: str) -> np.ndarray:
    """Separate visual QC for the rectified area used for F1; no silent crop."""
    roi_img = photo.copy()
    pts = np.rint(corners).astype(np.int32)
    cv2.polylines(roi_img, [pts], True, (255, 215, 55), max(2, round(min(photo.shape[:2])*.003)))
    full = fit_image_to_square(roi_img, size)
    rectified = photo_rect.copy()
    gt_overlay = overlay_scoring_gt(photo_rect, gt_rect)
    labels = [('Full photo + ROI outline', full),
              ('Rectified ROI used for F1', rectified),
              ('SAME scoring GT over rectified ROI', gt_overlay)]
    title_h, gap = 40, 8
    canvas = np.full((size+title_h+2*gap, 3*size+4*gap, 3), 245, dtype=np.uint8)
    for ix, (title, pic) in enumerate(labels):
        x = gap + ix*(size+gap)
        canvas[gap:gap+title_h,x:x+size] = (20,24,30)
        cv2.putText(canvas, title, (x+10,gap+26), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255,255,255), 1, cv2.LINE_AA)
        canvas[gap+title_h:gap+title_h+size, x:x+size] = pic
    return canvas


def results_for_one(spec:dict,base:Path,settings:dict,output_dir:Path,
                    mode:str,buffer_fraction:float,write_qc:bool,
                    red_threshold_override:float|None=None) -> list[dict[str,Any]]:
    sid=spec["id"]
    size=int(settings["rectified_size"])
    grid_n=int(settings["grid_size"])
    if size%grid_n:raise ValueError("rectified_size must be divisible by grid_size")
    brightness_min=int(settings["colour_brightness_min"])
    red_frac=float(settings["predicted_damage_red_fraction"] if red_threshold_override is None else red_threshold_override)
    gt_frac=float(settings["gt_damage_area_fraction_per_cell"])
    minpix=int(settings["min_classified_pixels_per_cell"])
    photo_path = resolve_input_image(base, spec, "ground_truth_photo")
    only_path = resolve_input_image(base, spec, "canupo_only")
    rc_path = resolve_input_image(base, spec, "roughness_canupo")
    photo = read_rgb(photo_path)
    only = read_rgb(only_path)
    rc = read_rgb(rc_path)
    scaled_photo_corners, aspect_error = resolve_photo_corners(spec, photo)
    # Use the scaled photo coordinates consistently for rectification and the
    # full-photo reference overlay. Do not modify original configuration values.
    effective_spec = dict(spec)
    effective_spec["photo_corners_xy"] = scaled_photo_corners
    # Construct ONE reference mask and reuse it for scoring and both QC views.
    photo_rect,gt_mask,poly_mask=make_gt_mask(photo,effective_spec,size,buffer_fraction,mode)
    marked_preview, gt_preview = make_preview_panels(photo, gt_mask, scaled_photo_corners, size)
    if aspect_error > 5.0:
        print(f"WARNING {sid}: photo aspect ratio differs from the stored reference "
              f"by {aspect_error:.1f}%. Inspect {sid}_{mode}_alignment_check.png "
              "before interpreting F1 scores. A different crop/view cannot be "
              "fixed by coordinate scaling alone.")
    only_corners=spec.get("canupo_only_corners_xy") or bbox_from_colours(only,brightness_min)
    rc_corners=spec.get("roughness_canupo_corners_xy") or bbox_from_colours(rc,brightness_min)
    ro,bo,_=warp_predictions(only,only_corners,size,brightness_min)
    rr,br,_=warp_predictions(rc,rc_corners,size,brightness_min)
    gt_cells,p_only,frac_o=binary_cell_labels(gt_mask,ro,bo,grid_n,gt_frac,red_frac,minpix)
    gt_cells_rc,p_rc,frac_rc=binary_cell_labels(gt_mask,rr,br,grid_n,gt_frac,red_frac,minpix)
    assert np.array_equal(gt_cells,gt_cells_rc)
    cnt_only=compute_counts(gt_cells,p_only)
    cnt_rc=compute_counts(gt_cells,p_rc)
    records=[]
    for model,cnt,red,blue,pred in [('CANUPO Only',cnt_only,ro,bo,p_only),('Roughness-CANUPO',cnt_rc,rr,br,p_rc)]:
        record={'specimen':sid, 'label_mode':mode, 'model':model,
                'selected_lnr_m':spec.get('selected_lnr_m',''),
                'grid_size':grid_n, 'total_cells':grid_n*grid_n,
                'gt_positive_cells':int(gt_cells.sum()),
                'gt_buffer_fraction_of_height':buffer_fraction,
                'red_cell_threshold':red_frac,
                'positive_predicted_cells':int(pred.sum()),
                'classified_pixels':int(np.count_nonzero(red)+np.count_nonzero(blue)),
                'red_pixels':int(np.count_nonzero(red)),
                'blue_pixels':int(np.count_nonzero(blue)),
                'canupo_bbox_used':json.dumps(only_corners if model=='CANUPO Only' else rc_corners),
                'photo_size_wh':json.dumps([int(photo.shape[1]), int(photo.shape[0])]),
                'photo_reference_size_wh':json.dumps(spec.get('photo_reference_size_wh')),
                'photo_aspect_ratio_error_percent':round(aspect_error, 2),
                'photo_roi_used':json.dumps(scaled_photo_corners),
                'annotation_status':'PROVISIONAL: manual spall polygons + approximate image registration' if mode=='damage' else 'PROVISIONAL: annotated magenta strokes, approximate image registration'}
        record.update(cnt.as_dict())
        records.append(record)
    if write_qc:
        vis=render_diagnostic(marked_preview,gt_preview,ro,bo,p_only,rr,br,p_rc,gt_cells,sid,mode)
        save_rgb(output_dir/f'{sid}_{mode}_review.png',vis)
        alignment=render_alignment_check(photo,photo_rect,gt_mask,scaled_photo_corners,size,sid)
        save_rgb(output_dir/f'{sid}_{mode}_alignment_check.png',alignment)
        # Raw binary mask is the exact 600x600 GT used for F1 and both QC views.
        raw_gt = np.repeat((gt_mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
        save_rgb(output_dir/f'{sid}_{mode}_GT_mask_rectified.png',raw_gt)
        np.savez_compressed(output_dir/f'{sid}_{mode}_grid_labels.npz',
                            gt=gt_cells,canupo_only=p_only,roughness_canupo=p_rc,
                            only_red_fraction=frac_o,rc_red_fraction=frac_rc)
    return records


def write_csv(path:Path, rows:list[dict[str,Any]])->None:
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(records:list[dict[str,Any]])->list[dict[str,Any]]:
    index={(r['specimen'],r['label_mode'],r['model']):r for r in records}
    summaries=[]
    for sid,mode in sorted({(r['specimen'],r['label_mode']) for r in records}):
        base=index[(sid,mode,'CANUPO Only')]
        rc=index[(sid,mode,'Roughness-CANUPO')]
        summaries.append({
            'specimen':sid,'label_mode':mode,'selected_lnr_m':base['selected_lnr_m'],
            'only_f1_percent':base['f1_percent'], 'rc_f1_percent':rc['f1_percent'],
            'change_percentage_points':round(rc['f1_percent']-base['f1_percent'],2),
            'only_TP':base['TP'],'only_FP':base['FP'],'only_FN':base['FN'],
            'rc_TP':rc['TP'],'rc_FP':rc['FP'],'rc_FN':rc['FN'],
            'gt_positive_cells':base['gt_positive_cells'],
            'status':'PRELIMINARY - not independent validated accuracy'
        })
    return summaries


def run(config_path:Path,base:Path,output_dir:Path,
        modes:list[str],save_qc:bool=True,do_sensitivity:bool=True)->None:
    config=json.loads(config_path.read_text(encoding='utf-8'))
    settings=config['settings']
    if not config.get('samples'):
        raise ValueError('No samples in configuration')
    output_dir.mkdir(parents=True,exist_ok=True)
    buffer=float(settings['gt_buffer_fraction_of_height'])
    print(f'Image input directory: {base.resolve()}')
    for spec in config['samples']:
        print(f"{spec['id']} input names: "
              f"GT={resolve_input_image(base, spec, 'ground_truth_photo').name}; "
              f"Only={resolve_input_image(base, spec, 'canupo_only').name}; "
              f"R-C={resolve_input_image(base, spec, 'roughness_canupo').name}")
    records=[]
    for spec in config['samples']:
        for mode in modes:
            rec=results_for_one(spec,base,settings,output_dir,mode,buffer,save_qc)
            records.extend(rec)
            summary=make_summary(rec)[0]
            print(f"{summary['specimen']} {mode:<6}: CANUPO {summary['only_f1_percent']:6.2f}%  "
                  f"R-C {summary['rc_f1_percent']:6.2f}%  "
                  f"delta {summary['change_percentage_points']:+6.2f} pp")
    write_csv(output_dir/'per_model_metrics.csv',records)
    write_csv(output_dir/'f1_comparison.csv',make_summary(records))
    sensitivity=[]
    if do_sensitivity:
        for spec in config['samples']:
            for mode in modes:
                for buf in config.get('sensitivity_buffer_fractions',[.02,.03,.05]):
                    for redfrac in config.get('sensitivity_red_fractions',[.20,.25,.30]):
                        test=results_for_one(spec,base,settings,output_dir,mode,float(buf),False,float(redfrac))
                        for t in test:
                            sensitivity.append({'specimen':spec['id'],'label_mode':mode,
                               'gt_buffer_fraction':float(buf),
                               'red_cell_threshold':float(redfrac),
                               'model':t['model'],'TP':t['TP'],'FP':t['FP'],'FN':t['FN'],
                               'F1_percent':t['f1_percent']})
        write_csv(output_dir/'parameter_sensitivity.csv',sensitivity)
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    (output_dir/'RUN_METADATA.json').write_text(json.dumps({
      'config_path':str(config_path),'config_sha256':config_hash,
      'data_directory':str(base),'label_modes':modes,
      'qc_gt_consistency':'Both QC overlays derive from the exact rectified F1 scoring GT mask; full-photo QC uses inverse-perspective projection.',
      'research_status':'PROVISIONAL; manual damage labeling and approximate image registration',
      'cautions':[
          'No independent held-out dataset, so scores are not proof of generalisation.',
          'Manually estimated photo bounding corners/spall regions require review.',
          'Full-size photo review does NOT imply accurate photo-to-cloud registration: check alignment images.',
          'Photo ROI coordinates are scaled by reference image size; new crops or viewpoints require manual registration.',
          'Image-based CANUPO outputs are NOT per-point class labels; black gaps are not counted as damage.',
          'Do not adjust thresholds after inspecting model performance to force an improvement.',
          'Damage and line-only versions are both reported to show sensitivity to the label definition.'
      ]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Outputs saved: {output_dir.resolve()}')


def main()->None:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path(__file__).resolve().parent
    p.add_argument('--config',type=Path,default=home/'CV7001_F1_study_config_PRIVATE.json',
                   help='study configuration JSON (default: next to this Python file)')
    p.add_argument('--input-dir',type=Path,default=home/'CV7001_F1_input',
                   help='folder of 12 images (S01, S01_CANUPO, S01_RC, ...), any supported extension')
    p.add_argument('--output-dir',type=Path,default=home/'CV7001_F1_results',
                   help='where results, CSV and visual checks are saved')
    p.add_argument('--label-mode',choices=['both','line','damage'],default='damage',
                   help='damage includes provisional spalling interiors; default: damage')
    p.add_argument('--no-sensitivity',action='store_true')
    p.add_argument('--no-qc',action='store_true')
    args=p.parse_args()
    modes=['line','damage'] if args.label_mode=='both' else [args.label_mode]
    run(args.config,args.input_dir,args.output_dir,modes,
        save_qc=not args.no_qc,do_sensitivity=not args.no_sensitivity)

if __name__=='__main__':
    main()
