import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import load_data, train_model

# Page config and title
st.set_page_config(page_title="🌸 Iris Species Classifier", layout="centered")
st.title("🌸 Iris Flower Species Classification")
st.markdown("Use ML models to classify iris species based on petal & sepal measurements.")

# Load data
df, iris = load_data()

# Dataset preview
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Pairplot
if st.checkbox("Show Pairplot"):
    fig = sns.pairplot(df, hue="species", palette="husl")
    st.pyplot(fig)

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])

# Train and evaluate
if st.button("Train and Evaluate"):
    model, score, report, conf_matrix, iris, fpr, tpr, roc_auc, avg_cv_score = train_model(model_choice)

    st.success(f"Model Accuracy: {score:.2f}")
    st.info(f"Average Cross-Validation Score (CV=5): {avg_cv_score:.2f}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=1))

    st.subheader("Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    st.subheader("Multiclass ROC Curve")
    fig2, ax2 = plt.subplots()
    for i, class_name in enumerate(iris.target_names):
        ax2.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve (One-vs-Rest)")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# Prediction UI
st.subheader("🔍 Predict Species")
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    model, _, _, _, iris, _, _, _, _ = train_model(model_choice)
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    st.info(f"🌼 Predicted Species: **{iris.target_names[prediction[0]]}**")
