# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Load the student placement dataset.
3. Preprocess the data: remove irrelevant columns and convert categorical data into numeric form.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression model using the training set.
6. Test the model with the testing set and evaluate performance (accuracy, confusion matrix).
7. Predict placement status for new student details.
8. End 

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Mahalakshmi B

RegisterNumber: 212224040182
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("Placement_Data.csv")
data = data.drop(["sl_no", "salary"], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop("status_Placed", axis=1)   
y = data["status_Placed"]                
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ["Not Placed", "Placed"]

plt.figure()
plt.imshow(cm, cmap="Reds")
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")
plt.tight_layout()
plt.show()
print("\n----- Enter New Student Details -----")
ssc_p = float(input("Enter SSC %: "))
hsc_p = float(input("Enter HSC %: "))
degree_p = float(input("Enter Degree %: "))
mba_p = float(input("Enter MBA %: "))
workex = input("Work Experience (Yes/No): ")
degree_t = input("Degree Type (Sci&Tech/Comm&Mgmt/Others): ")
specialisation = input("MBA Specialisation (Mkt&Fin/Mkt&HR): ")

new_data = {
    "ssc_p": [ssc_p],
    "hsc_p": [hsc_p],
    "degree_p": [degree_p],
    "mba_p": [mba_p],    
    "degree_t_Sci&Tech": [1 if degree_t == "Sci&Tech" else 0],
    "degree_t_Others": [1 if degree_t == "Others" else 0],
    "workex_Yes": [1 if workex == "Yes" else 0],
    "specialisation_Mkt&HR": [1 if specialisation == "Mkt&HR" else 0]
}

new_df = pd.DataFrame(new_data)
new_df = new_df.reindex(columns=X.columns, fill_value=0)
prediction = model.predict(new_df)
print("\nPrediction for the student:", 
      "Placed ✅" if prediction[0]==1 else "Not Placed ❌")
```
## Output:

<img width="677" height="752" alt="image" src="https://github.com/user-attachments/assets/48c31d70-5b97-4b44-a555-9fc6b3607437" />
<img width="687" height="243" alt="image" src="https://github.com/user-attachments/assets/a23042e8-58d8-448d-a802-9369a20875d7" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
