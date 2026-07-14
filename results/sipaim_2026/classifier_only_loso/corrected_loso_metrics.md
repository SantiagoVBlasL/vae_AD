# Corrected Frozen-Representation LOSO Metrics

| arm              |   target_site | input_lineage              |   n |   n_cn |   n_ad |   tn |   fp |   fn |   tp |   sensitivity |   specificity |   balanced_accuracy |       f1 |   predicted_ad_rate |      auc |   pr_auc |
|:-----------------|--------------:|:---------------------------|----:|-------:|-------:|-----:|-----:|-----:|-----:|--------------:|--------------:|--------------------:|---------:|--------------------:|---------:|---------:|
| locked           |           130 | locked_original            |  34 |     21 |     13 |   19 |    2 |    7 |    6 |      0.461538 |      0.904762 |            0.68315  | 0.571429 |            0.235294 | 0.835165 | 0.749378 |
| locked           |           035 | locked_original            |  20 |     15 |      5 |   14 |    1 |    3 |    2 |      0.4      |      0.933333 |            0.666667 | 0.5      |            0.15     | 0.56     | 0.405556 |
| corrected_combat |           130 | corrected_harmonized_input |  34 |     21 |     13 |   19 |    2 |    7 |    6 |      0.461538 |      0.904762 |            0.68315  | 0.571429 |            0.235294 | 0.798535 | 0.727903 |
| corrected_combat |           035 | corrected_harmonized_input |  20 |     15 |      5 |   10 |    5 |    3 |    2 |      0.4      |      0.666667 |            0.533333 | 0.333333 |            0.35     | 0.64     | 0.387967 |
