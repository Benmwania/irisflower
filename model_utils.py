from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris

def train_model(model_name):
    df, iris = load_data()
    X = df[iris.feature_names]
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    model = models[model_name]
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=iris.target_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    avg_cv_score = np.mean(cv_scores)

    # ROC & AUC (One-vs-Rest for multiclass)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_score = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(iris.target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return model, score, report, conf_matrix, iris, fpr, tpr, roc_auc, avg_cv_score
