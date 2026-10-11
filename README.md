# Zomato Restaurant Rating Prediction

Predicting a restaurant's aggregate rating from the Zomato restaurant dataset using scikit-learn regression models, with hyperparameter tuning, permutation-importance analysis, and an ablation study on the role of popularity (votes).

## Highlights

- Compared **Linear Regression**, **Random Forest**, and **Gradient Boosting**, tuned with `RandomizedSearchCV` (5-fold CV).
- Engineered features: log-scaled votes, cost relative to the currency median, number of cuisines, multi-hot encoded cuisines, city, and coordinates.
- Ran an **ablation without votes** to separate popularity from intrinsic signals (location, cuisine, price).
- Used **permutation importance** (instead of impurity importance) and a per-city **error analysis**.

## Results

| Setting | Model | Test R² | MAE |
|---|---|---|---|
| With votes | Linear Regression | 0.605 | 0.255 |
| With votes | Random Forest | 0.633 | 0.244 |
| With votes | Gradient Boosting | 0.650 | 0.240 |
| With votes | Gradient Boosting (tuned) | **0.654** | **0.240** |
| Without votes | Linear Regression | 0.437 | 0.328 |
| Without votes | Random Forest | 0.534 | 0.291 |
| Without votes | Gradient Boosting | 0.521 | 0.299 |
| Without votes | Gradient Boosting (tuned) | **0.533** | **0.293** |

Best tuned CV R²: **0.633** with votes, **0.542** without votes.

## Key findings

1. **Popularity explains a lot, but not everything.** Removing `log votes` drops R² from about 0.65 to about 0.53. Location, cuisine, and price alone still explain over half of the variance.
2. **Permutation importance shifts when votes are removed.**
   - With votes: `log votes` (0.60), `Cuisines` (0.11), `City` (0.11).
   - Without votes: `City` (0.43), `Cuisines` (0.14), `Longitude` (0.11), `Relative cost` (0.10). Location absorbs part of the role votes played.
3. **Errors are spread evenly.** In the cities with enough data (New Delhi, Gurgaon, Noida), mean absolute error is about 0.29 to 0.31, close to the overall MAE. No city is an outlier.
4. **Model plateau.** Extra tuning and added features moved CV R² by only about 0.01, so the ceiling appears to come from the data (it contains no direct measure of food quality, service, or review text).

## Limitations

- **Geographic concentration:** most restaurants are in the Delhi NCR area (New Delhi, Gurgaon, Noida, Faridabad); other cities have only about 20 restaurants each, so results there are not statistically reliable.
- **Votes are partly an outcome of the rating:** well-rated places attract more reviews. The with-votes model predicts the rating of an *existing, reviewed* restaurant; the without-votes model is closer to predicting a *new* restaurant.
- Rows with rating 0 (unrated) and missing coordinates were removed.

## Project structure

```
.
├── zomato_rating_prediction.py
├── requirements.txt
└── README.md
```

## Dataset

Zomato Restaurants dataset (download from Kaggle: _add link here_). Place the CSV file named `Zomato Restaurant Dataset.csv` in the same folder as the script.

## How to run

```bash
pip install -r requirements.txt
python zomato_rating_prediction.py
```

To reproduce the **without votes** experiment, remove `"log votes"` from `num_cols` in the script and run it again.

## Tech stack

Python, pandas, NumPy, scikit-learn
