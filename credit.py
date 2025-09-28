

import pandas as pd

df=pd.read_csv('/content/credit_card_default.csv')

df.head(10)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X = df.drop(["default payment_next_month", "BILL_AMT6"], axis=1)
y = df["default payment_next_month"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(df.columns)

from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

X = df.drop(["default payment_next_month", "BILL_AMT6"], axis=1)
y = df["default payment_next_month"]


selected_features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

n = df.iloc[[3]]
if "default payment_next_month" in n.columns:
    n = n.drop("default payment_next_month", axis=1)
n = n.reindex(columns=selected_features, fill_value=0)
pred = lr.predict(n.values)
prob = lr.predict_proba(n.values)[:, 1]

print("Prediction:", pred)
print("Probability of Default:", prob)

row_idx = 4
n = df.iloc[[row_idx]]
if "default payment_next_month" in n.columns:
    n = n.drop("default payment_next_month", axis=1)

n = n.reindex(columns=selected_features, fill_value=0)
lr_pred = lr.predict(n.values)[0]

rf_pred = rf.predict(n.values)[0]


if lr_pred == 0 and rf_pred == 0:
    final_answer = "YES for eligible"
else:
    final_answer = "NO"

print(" Credit Scoring Decision")
print("--------------------------------------------------------------------------------------------")
print(f"Logistic Regression: {'Good' if lr_pred==0 else 'Bad'}")
print(f"Random Forest: {'Good' if rf_pred==0 else 'Bad'}")
print("--------------------------------------------------------------------------------------------")
print(f"✅ Final Answer: {final_answer}")

