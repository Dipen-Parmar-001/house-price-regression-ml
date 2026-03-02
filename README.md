🏠 House Price Prediction using Random Forest

An end-to-end machine learning project that predicts residential property prices using the King County housing dataset. The model is built with a Random Forest Regressor, includes feature engineering, and provides a reusable prediction pipeline for real-time price estimation.


---

📌 Project Overview

This project demonstrates a complete ML workflow:

Data loading and preprocessing

Feature engineering

Model training with Random Forest

Model evaluation (R² and RMSE)

Model serialization with joblib

Production-style prediction pipeline


The goal is to build a reliable baseline model for house price estimation.


---

🗂️ Repository Structure

house-price-prediction-random-forest/
│
├── data/
│   └── Housing.csv
│
├── artifacts/
│   ├── model.pkl        # (generated locally)
│   └── columns.pkl
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md


---

⚙️ Features Used

Key input features include:

bedrooms

bathrooms

sqft_living

sqft_lot

floors

waterfront

view

condition

grade

zipcode

latitude & longitude

etc.


🔧 Engineered Features

house_age = 2026 − yr_built

is_renovated = whether house was renovated



---

🤖 Model

Algorithm: Random Forest Regressor

Trees: 600

Parallelism: n_jobs = -1

Validation: Out-of-Bag (OOB) + Test split



---

📊 Model Performance

(Update numbers if yours differ)

OOB R² : ~0.877
Test R²: ~0.87
RMSE   : ~147,000

These results indicate strong baseline performance for this dataset.


---

🚀 How to Run

1️⃣ Clone the repository

git clone https://github.com/<your-username>/house-price-prediction-random-forest.git
cd house-price-prediction-random-forest


---

2️⃣ Install dependencies

pip install -r requirements.txt


---

3️⃣ Train the model

python train.py

This will generate:

artifacts/model.pkl

artifacts/columns.pkl



---

4️⃣ Run prediction

python predict.py

You can modify the sample input inside predict.py.


---

🧠 Example Prediction

sample_house = {
    "bedrooms": 4,
    "bathrooms": 2.5,
    "sqft_living": 2350,
    ...
}

Output:

Predicted Price: 577,365


---

⚠️ Note on Model File

The trained model file may be excluded from the repository due to GitHub size limits.
If missing, simply run:

python train.py

to regenerate the model locally.


---

🛠️ Tech Stack

Python

pandas

NumPy

scikit-learn

joblib



---

📈 Future Improvements

Possible enhancements:

Gradient Boosting / XGBoost

Hyperparameter tuning

Outlier handling

Feature importance visualization

Web app deployment (Streamlit/FastAPI)



---

👤 Author

Dipen Parmar


---

⭐ If you found this project useful, consider giving it a star!
