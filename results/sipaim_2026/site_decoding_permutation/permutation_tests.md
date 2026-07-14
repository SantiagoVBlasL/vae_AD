# Permutation Tests

**N permutations:** 1000. **Seed:** 42.

## Per-fold results

|   fold | test                                |   observed |   perm_mean |   perm_sd |   perm_p |   n_perm |
|-------:|:------------------------------------|-----------:|------------:|----------:|---------:|---------:|
|      1 | knn5_manufacturer_balanced_accuracy |     0.6007 |      0.3297 |    0.0505 |   0.0010 |     1000 |
|      1 | centroid_dist_GE_vs_Philips         |     6.5301 |      3.9690 |    0.3808 |   0.0010 |     1000 |
|      2 | knn5_manufacturer_balanced_accuracy |     0.6366 |      0.3347 |    0.0535 |   0.0010 |     1000 |
|      2 | centroid_dist_GE_vs_Philips         |     5.8541 |      3.8344 |    0.3802 |   0.0010 |     1000 |
|      3 | knn5_manufacturer_balanced_accuracy |     0.6899 |      0.3348 |    0.0516 |   0.0010 |     1000 |
|      3 | centroid_dist_GE_vs_Philips         |     6.5606 |      4.1052 |    0.4219 |   0.0010 |     1000 |
|      4 | knn5_manufacturer_balanced_accuracy |     0.7334 |      0.3319 |    0.0519 |   0.0010 |     1000 |
|      4 | centroid_dist_GE_vs_Philips         |     6.5774 |      4.0699 |    0.4118 |   0.0010 |     1000 |
|      5 | knn5_manufacturer_balanced_accuracy |     0.6687 |      0.3308 |    0.0527 |   0.0010 |     1000 |
|      5 | centroid_dist_GE_vs_Philips         |     6.1582 |      3.8059 |    0.3642 |   0.0010 |     1000 |

## Summary across folds

| test                                |   obs_mean |   obs_sd |   perm_p_mean |   perm_p_min |   perm_p_max |
|:------------------------------------|-----------:|---------:|--------------:|-------------:|-------------:|
| centroid_dist_GE_vs_Philips         |     6.3361 |   0.3202 |        0.0010 |       0.0010 |       0.0010 |
| knn5_manufacturer_balanced_accuracy |     0.6659 |   0.0506 |        0.0010 |       0.0010 |       0.0010 |

