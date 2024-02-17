import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('out_our.csv')

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
X_eval = pd.read_csv('out_rela-0.csv').drop('Label', axis=1)

# List of models
models = [
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('MLPClassifier', MLPClassifier(random_state=42)),
    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),
    ('SVC', svm.SVC(probability=True, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('GaussianNB', GaussianNB())
]

# Train each model, print the accuracy, and print the mean of the evaluation probabilities
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Accuracy:', accuracy_score(y_test, y_pred))
    
    probabilities = model.predict_proba(X_eval)
    focus_avg = probabilities[:, 1::2].flatten().mean()
    print(f'{name} Focus Average:', focus_avg)