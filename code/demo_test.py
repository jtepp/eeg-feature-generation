import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/original_data/subject.csv')

# load the model from disk
filename = 'random_forest_model.pkl'
model = pickle.load(open(filename, 'rb'))

input = X_test.iloc[0]
input = input.values.reshape(1, -1)
result = model.predict(input)

print(result, y_test.iloc[0])