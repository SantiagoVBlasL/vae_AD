# Latent Site/Manufacturer Geometry Report

**Run:** recover035_latent384_beta3p75_T80_h10000_p560_full5x5
**Analysis date:** 2026-07-01
**Script:** run_latent_site_geometry.py
**N subjects (OOF test):** 397
**Latent dim:** 384
**UMAP:** Computed (umap-learn 0.5.3)

---

## 1. Silhouette scores (Euclidean, up to 500 subjects subsample)

| Grouping | n_classes | Silhouette |
|---|---:|---:|
| Diagnosis (AD/CN) | 2 | -0.0050 |
| Manufacturer (GE/Philips/SIEMENS) | 3 | 0.0011 |
| Site (Site3) | 35 | -0.0954 |

**Interpretation:** Silhouette > 0.1 indicates detectable clustering;
values near 0 indicate no structure; negative values indicate misassignment.

---

## 2. kNN decodability (k=5, balanced accuracy, 3-fold stratified CV)

| target       |   n_classes |   balanced_accuracy_mean |   balanced_accuracy_std |
|:-------------|------------:|-------------------------:|------------------------:|
| manufacturer |           3 |                 0.616971 |              0.0219162  |
| diagnosis    |           2 |                 0.512285 |              0.00944969 |
| site_top10   |           9 |                 0.106983 |              0.00442764 |

**Key question:** Is manufacturer clustering stronger than diagnosis clustering?

---

## 3. Manufacturer centroid pairwise distances

| manufacturer_i   | manufacturer_j   |   euclidean_dist |   cosine_dist |
|:-----------------|:-----------------|-----------------:|--------------:|
| GE               | Philips          |          2.8205  |      0.471006 |
| GE               | SIEMENS          |          2.55598 |      0.384449 |
| Philips          | SIEMENS          |          2.52932 |      0.415766 |

---

## 4. Manufacturer within-group radius (mean Euclidean distance to centroid)

| group   |   n |   within_radius_mean |
|:--------|----:|---------------------:|
| GE      | 122 |              14.7718 |
| Philips | 145 |              14.6512 |
| SIEMENS | 130 |              14.582  |

---

## 5. Prediction error breakdown

| Type | N |
|---|---:|
| TP | 21 |
| TN | 291 |
| FP | 9 |
| FN | 76 |

**FP by manufacturer:**
Manufacturer
Philips    8
GE         1

**FN by manufacturer:**
Manufacturer
Philips    33
SIEMENS    23
GE         20

**Top sites by FP+FN count:**
  - Site 168: 6
  - Site 130: 6
  - Site 19: 5
  - Site 135: 5
  - Site 6: 4

---

## 6. Error type vs centroid distance

| Manufacturer   |   correct |   mean_dist |   median_dist |   std_dist |   n |
|:---------------|----------:|------------:|--------------:|-----------:|----:|
| GE             |         0 |     14.5126 |       14.7733 |   0.79438  |  21 |
| GE             |         1 |     14.8257 |       14.8037 |   0.752954 | 101 |
| Philips        |         0 |     14.4625 |       14.5596 |   0.678067 |  41 |
| Philips        |         1 |     14.7257 |       14.6971 |   0.729902 | 104 |
| SIEMENS        |         0 |     14.2294 |       14.3818 |   0.7788   |  23 |
| SIEMENS        |         1 |     14.6578 |       14.6687 |   0.750292 | 107 |

---

## 7. PCA variance explained

| PC | Explained variance ratio |
|---|---:|
| PC1 | 0.0360 |
| PC2 | 0.0331 |
| PC3 | 0.0293 |
| PC4 | 0.0281 |
| PC5 | 0.0177 |
| top-10 cumulative | 0.2244 |

---

## 8. Figures generated

- `figures/pca_pc1_pc2_diagnosis_manufacturer.pdf`
- `figures/pca_pc1_pc2_error_type.pdf`
- `figures/umap_diagnosis_manufacturer.pdf` (if UMAP available)
- `figures/umap_error_type.pdf` (if UMAP available)
- `figures/umap_site.pdf` (if UMAP available)
- `figures/centroid_distance_by_error_type.pdf`
- `figures/silhouette_per_sample.pdf`
- `figures/knn_decodability.pdf`

---

## 9. Interpretation summary

A high manufacturer silhouette combined with low diagnosis silhouette would
confirm that the latent space encodes scanner type more strongly than
pathology — a site-confound risk. Values close to zero for both suggest
the β-VAE has collapsed both signals equally.

The kNN balanced accuracy comparison (diagnosis vs manufacturer) is the
most direct test: if `knn(manufacturer) >> knn(diagnosis)`, the latent
space is scanner-dominated.

FP/FN concentration in specific manufacturer or site clusters would
indicate that classification errors are spatially structured in the latent
space — a sign that the classifier's errors track hardware/site factors
rather than random noise.

---

## Caveats

- Analysis uses OOF test-set latents only (n=397); training-set geometry may differ.
- Site3 has 35 unique values; silhouette computed on all but kNN
  filtered to sites with ≥10 subjects to avoid degenerate CV folds.
- Silhouette uses Euclidean distance in 384-dim space (no dimensionality reduction).
- Mapper/persistent homology: skipped (not requested unless dependencies confirmed).
