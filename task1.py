import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load dataset
data = pd.read_csv("german_credit_data.csv")

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Convert categorical to numeric
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop("Risk_bad", axis=1)
y = data["Risk_bad"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate models
models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print("\n", name)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# Save best model
joblib.dump(rf, "credit_scoring_model.pkl")

print("\nModel saved as credit_scoring_model.pkl")
