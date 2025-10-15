import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import kagglehub
import os

# ---------------------------
# Load dataset (via kagglehub)
# ---------------------------
st.title("💳 Credit Card Fraud Detection App")

st.write("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")

data = pd.read_csv(csv_path)
st.success("✅ Dataset loaded successfully!")

# ---------------------------
# Preprocessing
# ---------------------------
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balance dataset (undersampling)
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=2
)

# ---------------------------
# Train model (once)
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.write(f"🔹 Training Accuracy: **{train_acc:.3f}**")
st.write(f"🔹 Testing Accuracy: **{test_acc:.3f}**")

# ---------------------------
# Prediction section
# ---------------------------
st.header("🔍 Predict Transaction Type")

st.write("Enter 30 feature values (comma-separated) like:")
st.code("0.0, -1.3598, -0.0727, 2.5363, 1.3781, ... , 149.62")

user_input = st.text_input("Enter all 30 features:")
submit = st.button("Submit")

if submit:
    try:
        # Parse input
        features = np.array(user_input.split(","), dtype=np.float64).reshape(1, -1)

        if features.shape[1] != X.shape[1]:
            st.error(f"❌ Expected {X.shape[1]} features, got {features.shape[1]}")
        else:
            # Scale using same scaler
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            if prediction[0] == 0:
                st.success("✅ Legitimate Transaction")
            else:
                st.error("⚠️ Fraudulent Transaction Detected!")
    except Exception as e:
        st.error(f"Invalid input: {e}")

