# credit-card-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/shiva/OneDrive/Desktop/internship encryptix/credit card/creditcard.csv")

print("Dataset Overview:")
print("-----------------")
print("Shape:", data.shape)
print("Summary Statistics:")
print(data.describe())
print("Class Distribution:")
print(data['Class'].value_counts())

X = data.drop('Class', axis=1)  
y = data['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("-----------------")
print("Model: Logistic Regression Classifier")
print("Accuracy: {:.3f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.3f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.3f}%".format(recall_score(y_test, y_pred) * 100))
print("F1-Score: {:.3f}%".format(f1_score(y_test, y_pred) * 100))
print("Matthews Correlation Coefficient: {:.3f}%".format(matthews_corrcoef(y_test, y_pred) * 100))

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix\nLogistic Regression Classifier")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

confusion_df = pd.DataFrame(conf_matrix, index=LABELS, columns=LABELS)
print("\nConfusion Matrix:")
print("-----------------")
print(confusion_df)

total_instances = sum(sum(conf_matrix))
correctly_classified = confusion_df.iloc[0, 0] + confusion_df.iloc[1, 1]
incorrectly_classified = total_instances - correctly_classified
print("\nClassification Results:")
print("-----------------------")
print("Correctly Classified Instances: {:.2f}% ({})".format(correctly_classified / total_instances * 100, correctly_classified))
print("Incorrectly Classified Instances: {:.2f}% ({})".format(incorrectly_classified / total_instances * 100, incorrectly_classified))
