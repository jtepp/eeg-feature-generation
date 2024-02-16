import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('out_relaxing.csv')

X = df.drop('Label', axis=1)
y = df['Label']



# load the model from disk
filename = 'good_model_RF.pkl'
model = pickle.load(open(filename, 'rb'))

input = X.iloc[0]
input = input.values.reshape(1, -1)
# result = model.predict(input)
probabilities = model.predict_proba(X)

# average every 2nd index of every array in this array
focus_avg = probabilities[:, 1::2].flatten().mean()
print(probabilities,focus_avg)