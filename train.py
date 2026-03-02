# Import Libraries

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

#Load Data From CSV-File

data = pd.read_csv('/content/Housing.csv')
df = pd.DataFrame(data)
df = df.drop(columns=['date', 'id'])

df['house_age'] = 2026 - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

model = RandomForestRegressor(
        n_estimators=600,
        oob_score=True,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
)

X = df.drop(columns=['price'])

joblib.dump(X.columns.tolist(), "columns.pkl")

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Training Completed. Model Saved.")
