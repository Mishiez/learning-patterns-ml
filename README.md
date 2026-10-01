# learning-patterns-ml
Random Forest vs baseline models predicting student dropout, plus K-Means clustering of African countries using World Bank development indicators, visualized with PCA.

## Objective

Two small machine-learning projects that show I can choose, evaluate and explain models:

1. **Supervised:** predict whether a university student will *Dropout*, stay *Enrolled* or *Graduate*, and test whether a Random Forest beats simple baselines.
2. **Unsupervised:** group African countries by development indicators with K-Means, choose the number of clusters with a stated method, and find out where Kenya lands.

## Datasets

| Part | Data | Source |
|---|---|---|
| 1 | Predict Students' Dropout and Academic Success (4,424 students, 36 features, 3 classes) | [UCI Machine Learning Repository, id 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), loaded with `ucimlrepo` |
| 2 | Eight development indicators for all 54 African countries, averaged over 2018-2022 | [World Bank Open Data](https://data.worldbank.org/), loaded with [`wbgapi`](https://pypi.org/project/wbgapi/) |

The World Bank indicators are: GDP per capita (`NY.GDP.PCAP.CD`), life expectancy (`SP.DYN.LE00.IN`), under-5 mortality (`SH.DYN.MORT`), fertility rate (`SP.DYN.TFRT.IN`), access to electricity (`EG.ELC.ACCS.ZS`), internet users (`IT.NET.USER.ZS`), urban population (`SP.URB.TOTL.IN.ZS`) and agriculture share of GDP (`NV.AGR.TOTL.ZS`).

"All 54" means the World Bank's 48 Sub-Saharan countries plus Algeria, Djibouti, Egypt, Libya, Morocco and Tunisia.

## Repository layout

```
├── README.md
├── requirements.txt
├── data/
│   ├── africa_indicators_2018_2022_raw.csv   raw country averages (before imputation)
│   └── model_comparison.csv                  final model comparison table
├── notebooks/
│   ├── 01_supervised_student_dropout.ipynb
│   └── 02_unsupervised_africa_indicators.ipynb
└── figures/                                  all saved plots
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd notebooks
jupyter lab        # open a notebook and choose Run All
```

Run the notebooks from inside the `notebooks/` folder, because they save figures and data using relative paths (`../figures`, `../data`). Both need an internet connection (the first downloads from UCI, the second from the World Bank). Everything uses `random_state=42`, and the first notebook takes a couple of minutes to run.

## Part 1 results: student dropout

The test set (885 students) was used once, at the end. Train and CV scores are from 5-fold cross-validation on the training set.

| Model | Test accuracy | Test macro F1 | Test Dropout recall | Train macro F1 | CV macro F1 |
|---|---|---|---|---|---|
| Dummy (most frequent class) | 0.499 | 0.222 | 0.000 | 0.222 | 0.222 |
| Logistic regression | 0.767 | 0.693 | 0.750 | 0.755 | 0.702 |
| Random Forest (default) | 0.767 | 0.692 | 0.746 | 1.000 | 0.685 |
| **Random Forest (tuned)** | 0.753 | **0.704** | 0.718 | 1.000 | 0.717 |
| Random Forest (tuned, without semester grades) | 0.593 | 0.548 | 0.627 | 0.923 | 0.586 |

## Part 2 results: African country clusters (k = 3)

| Cluster | Countries | GDP per capita | Life expectancy | Under-5 mortality | Electricity access | Internet users |
|---|---|---|---|---|---|---|
| **Higher-income, urbanised, connected** | 14 (North Africa, island states, Botswana, South Africa, Gabon, Equatorial Guinea) | $6,249 | 70.3 | 28 | 87% | 65% |
| **Middle group, developing** | 21 (includes **Kenya**, Tanzania, Uganda, Rwanda, Ghana, Senegal, Zambia) | $1,805 | 63.7 | 51 | 58% | 32% |
| **Lowest income, highest child mortality** | 19 (Sahel, Central Africa, Ethiopia, Mozambique, Nigeria) | $873 | 58.8 | 96 | 30% | 17% |

k = 3 was chosen from the elbow curve and silhouette scores for k = 2..8 (see the notebook for the reasoning).

## Key takeaways

- **The Random Forest clearly beats the naive baseline** (macro F1 0.704 vs 0.222), but beats logistic regression only slightly (0.704 vs 0.693). That gap is about the size of the fold-to-fold noise, so I treat the two as roughly tied. The forest's real advantage is on the hard "Enrolled" class.
- **Accuracy is misleading here.** Always predicting "Graduate" gives about 50% accuracy while finding no dropouts at all, so I judged models mainly by macro F1 and Dropout recall.
- **Bias-variance, with this model's numbers:** a single unconstrained tree scores 1.00 on training data but 0.627 in cross-validation. Averaging many such trees raises the cross-validated score to 0.685 (default forest) and 0.717 (tuned forest) by reducing variance. The tuning grid was very flat (0.708-0.717), so the forest is not sensitive to its settings.
- **Leakage matters.** The first- and second-semester grade columns are only known after enrolment. Removing them drops macro F1 from 0.704 to 0.548, and that lower number is the honest score for predicting at enrolment.
- **"Enrolled" is hardest** because it is the smallest class (18%) and a transition state between the other two outcomes.
- **Countries lie on a spectrum, not in sharp groups.** The silhouette score for k = 3 is only 0.265, and one "development level" axis carries 69% of the variance. The clusters are useful bands along that axis, and countries near the borders (for example Ethiopia, Togo, Nigeria) could move with a slightly different setup.
- **Kenya** falls in the middle group with its East African neighbours Tanzania, Uganda and Rwanda. It is better than that group's average on electricity, child mortality and fertility, and worse on urbanisation and internet use.
- **Limitations:** only 54 countries, five-year averaging, median-imputed values for four countries, and one data quirk (the Central African Republic's life-expectancy series looks unreliable). These are discussed in the notebooks.
