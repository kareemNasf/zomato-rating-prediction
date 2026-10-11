import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ----------
file_path = "Zomato Restaurant Dataset.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.info())
df =df[df['Aggregate rating'] > 0 ].copy()  # Filter out rows with Aggregate rating <= 0
df["Cuisines"] = df["Cuisines"].fillna("Unknown")
print(df['Aggregate rating'].value_counts())
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
print("="*20)
#==============================================================
df['Relative cost']= df['Average Cost for two'] / df.groupby('Currency')['Average Cost for two'].transform('median')
df['log votes'] = np.log1p(df['Votes'])
df["Num cuisines"] = df["Cuisines"].str.split(",").str.len()
for c in ["Has Table booking", "Has Online delivery"]:
    df[c] = (df[c] == "Yes").astype(int)
#==============================================================
num_cols = [ "Relative cost", "Num cuisines","Price range",
            "Has Table booking", "Has Online delivery", "Latitude", "Longitude"]
cat_cols = ["Country Code", "City"]

x = df[num_cols + cat_cols + ["Cuisines"]]
y = df["Aggregate rating"]
#================================train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ---------- Pipeline
def split_cuisines(s):
    return [c.strip() for c in s.split(",") if c.strip()]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), cat_cols),
    ("cuis", CountVectorizer(tokenizer=split_cuisines, token_pattern=None,
                             lowercase=False, binary=True, max_features=80), "Cuisines"),  # (3)
], remainder="passthrough", sparse_threshold=0)   # dense عشان HistGB
#-====================================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=500, n_jobs=-1, min_samples_leaf=3, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, m in models.items():
    pipe = Pipeline([("prep", prep), ("model", m)])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    cv = cross_val_score(pipe, x_train, y_train, cv=kf, scoring="r2").mean()
    print(f"{name}: R2={r2_score(y_test, pred):.3f} | MAE={mean_absolute_error(y_test, pred):.3f} | CV R2={cv:.3f}")   # <-- جوّا الـ loop
 #=====================================================================
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__n_estimators": [300, 500, 700],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_samples_leaf": [1, 3, 5, 10],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
}

pipe = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(random_state=42))])
search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=kf, scoring="r2", n_jobs=-1, random_state=42)
search.fit(x_train, y_train)

best_pred = search.predict(x_test)
print("Best params:", search.best_params_)
print("Best CV R2:", round(search.best_score_, 3))
print("Test R2:", round(r2_score(y_test, best_pred), 3), "| MAE:", round(mean_absolute_error(y_test, best_pred), 3))
# ---------- (5) Permutation importance على الأعمدة الأصلية
r = permutation_importance(search.best_estimator_, x_test, y_test,
                           n_repeats=10, random_state=42, scoring="r2")
print("\nPermutation importance:\n",
      pd.Series(r.importances_mean, index=x_test.columns).sort_values(ascending=False).round(3))
      #==========================================
res = (y_test - best_pred).abs()
print(x_test.assign(res=res).groupby("City")["res"]
      .agg(["mean", "count"]).query("count > 5").sort_values("mean").tail(10))

print(df["City"].value_counts().head(10))