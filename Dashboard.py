import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Secure Fraud Detection Dashboard", layout="wide")

# -----------------------------
# Session State
# -----------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}

# -----------------------------
# Sidebar Auth
# -----------------------------
with st.sidebar:
    st.header("🔐 Secure Access")

    st.subheader("Register")
    new_user = st.text_input("New Username")
    new_pwd = st.text_input("New Password", type="password")

    if st.button("Register"):
        if new_user and new_pwd:
            st.session_state.users[new_user] = new_pwd
            st.success("Registered successfully")
        else:
            st.error("Enter username and password")

    st.divider()

    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in st.session_state.users and st.session_state.users[user] == pwd:
            st.session_state.auth = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# Stop if not logged in
if not st.session_state.auth:
    st.warning("Please register/login to access dashboard")
    st.stop()

# -----------------------------
# Main Title
# -----------------------------
st.title("💳 Secure Financial Fraud Detection ")
st.write("Ensemble Learning + SMOTE + Security Based Fraud Detection")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df

try:
    df = load_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Faster training subset
    X_small = X_train[:50000]
    y_small = y_train[:50000]

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_small, y_small)

    # Models
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=50)
    dt = DecisionTreeClassifier()

    model = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("dt", dt)],
        voting="soft"
    )

    model.fit(X_res, y_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.3).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{(y_pred == y_test).mean()*100:.2f}%")
    c2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.2f}")
    c3.metric("Total Rows", len(df))
    c4.metric("Fraud Cases", int(y.sum()))

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Overview", "Graphs", "Live Prediction"])

    # Overview
    with tab1:
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head())

        st.subheader("Class Distribution")

        counts = df["Class"].value_counts()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie(
            counts,
            labels=["Normal", "Fraud"],
            autopct="%1.2f%%"
        )
        ax.set_title("Transaction Distribution")

        st.pyplot(fig)

        st.write("Class 0 (Normal):", counts[0])
        st.write("Class 1 (Fraud):", counts[1])

    # Graphs
    with tab2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        cm = confusion_matrix(y_test, y_pred)
        ax[0].imshow(cm)
        ax[0].set_title("Confusion Matrix")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax[0].text(j, i, cm[i, j], ha="center", va="center")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax[1].plot(fpr, tpr)
        ax[1].set_title("ROC Curve")

        st.pyplot(fig)

    # Prediction
    with tab3:
        st.subheader("🛡️ Live Fraud Prediction")

        st.write("Enter new transaction details:")

        input_data = []

        for col in X.columns:
            val = st.number_input(f"{col}", value=0.0, format="%.4f")
            input_data.append(val)

        if st.button("Predict New Transaction"):
            input_df = pd.DataFrame([input_data], columns=X.columns)

            input_scaled = scaler.transform(input_df)

            prob = model.predict_proba(input_scaled)[0][1]
            pred = "FRAUD" if prob > 0.3 else "NORMAL"

            st.success(f"Prediction: {pred}")
            st.info(f"Fraud Probability: {prob:.4f}")

            if pred == "FRAUD":
                st.error("⚠️ Security Alert: Suspicious transaction detected")

except Exception as e:
    st.error(f"Error: {e}")