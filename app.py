from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    train_acc = None
    test_acc = None
    cm = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            target_column = request.form["target"]

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X = X.select_dtypes(include=np.number)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=1
            )

            model = GaussianNB()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_acc = round(accuracy_score(y_train, y_pred_train), 4)
            test_acc = round(accuracy_score(y_test, y_pred_test), 4)

            cm = confusion_matrix(y_test, y_pred_test)

    return render_template("index.html",
                           train_acc=train_acc,
                           test_acc=test_acc,
                           cm=cm)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)