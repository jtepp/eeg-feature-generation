import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load the data
df = pd.read_csv('out_train.csv')

if 'Timestamp' in df.columns:
    df = df.drop(['Timestamp'], axis=1)

# Separate the first row
df_head = df.iloc[:1]

# Shuffle the rest
df_rest = df.iloc[1:].sample(frac=1, random_state=50)

# Concatenate the first row and the shuffled DataFrame
df = pd.concat([df_head, df_rest])


X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the RandomForestClassifier (100 is optimal n_estimators)
model = RandomForestClassifier(n_estimators=100, random_state=42) # 0.9539007092198581
# model = MLPClassifier(random_state=42) # 0.8262411347517731
# model = GradientBoostingClassifier(random_state=42) # 0.9822695035460993
# model = svm.SVC(probability=True, random_state=42) # 0.6631205673758865
# model = LogisticRegression(random_state=42) # 0.723404255319149
# model = KNeighborsClassifier() # 0.9574468085106383
# model = GaussianNB() # 0.5460992907801419



# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:')
print(cnf_matrix)

# Print the accuracy score
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the model
model_file = 'random_forest_model.pkl'
pickle.dump(model, open(model_file, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)