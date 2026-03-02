import pandas as pd
import joblib

# Load Model
model = joblib.load("model.pkl")

def predict_price(input_data: dict):
        df = pd.DataFrame([input_data])
        df['house_age'] = 2026 - df['yr_built']
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

        columns = joblib.load("columns.pkl")
        df = df.reindex(columns=columns)
        pred = model.predict(df)[0]
        return round(pred, 2)

if __name__ == "__main__":
       sample_house = {
        # MUST match with training columns
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2350,
        "sqft_lot": 6800,
        "floors": 2.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 8,
        "sqft_above": 2350,
        "sqft_basement": 0,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98103,
        "lat": 47.6750,
        "long": -122.123,
        "sqft_living15": 2200,
        "sqft_lot15": 7000
    }

       price = predict_price(sample_house)
       print("Predicted Price:", price)
