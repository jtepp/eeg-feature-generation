import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from pandas.plotting import scatter_matrix

df = pd.read_csv('out_schaffer.csv')

X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load the model from disk
filename = 'good_model_RF.pkl'
model = pickle.load(open(filename, 'rb'))

# Visualize the first tree from the forest
plt.figure(figsize=(20,10))
tree.plot_tree(model.estimators_[0], 
               feature_names=X.columns,
               class_names=["Relaxing", "Concentrating"],
               filled=True)

plt.savefig('tree.png', dpi=300)
plt.show()
plt.close()