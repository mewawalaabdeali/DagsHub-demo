import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub

dagshub.init(repo_owner='mewawalaabdeali', repo_name='DagsHub-demo', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/mewawalaabdeali/DagsHub-demo.mlflow')

#load the iris dataset
iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

max_depth = 20
#n_estimators = 75

mlflow.set_experiment("iris_dt")
#apply mlflow to train

with mlflow.start_run():
    #mlflow.log_param("max_depth", max_depth)
    #mlflow.log_param("n_estimators", n_estimators)

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    #mlflow.log_param("n_estimators", n_estimators)

    #log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    #save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, "decision tree")
    mlflow.set_tag('author', 'ali')
    mlflow.set_tag('model', 'decision tree')

    print('accuracy', accuracy)


