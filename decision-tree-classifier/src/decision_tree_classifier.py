import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data = pd.read_csv('bank-additional-full.csv', sep=';')

print(f"Dataset shape: {data.shape}")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

data = pd.get_dummies(data, drop_first=True)

X = data.drop('y_yes', axis=1)
y = data['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Step 6: Hyperparameter Tuning (Optional)
clf_tuned = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=10, random_state=42)
clf_tuned.fit(X_train, y_train)

# Evaluate the Tuned Model
y_tuned_pred = clf_tuned.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_tuned_pred)
print(f"Tuned Model Accuracy: {tuned_accuracy:.2f}")

# Step 7: Save the Model and Predictions
# Save the trained model
joblib.dump(clf_tuned, 'decision_tree_model.pkl')

# Export predictions to a CSV
predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_tuned_pred})
predictions.to_csv('predictions.csv', index=False)

print("Model saved as 'decision_tree_model.pkl' and predictions saved as 'predictions.csv'.")
