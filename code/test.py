import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('out_conc-0.csv')

X = df.drop('Label', axis=1)
# y = df['Label']



# load the model from disk
filename = 'random_forest_model.pkl'
model = pickle.load(open(filename, 'rb'))

probabilities = model.predict_proba(X)

# average every 2nd index of every array in this array
focus_avg = probabilities[:, 1::2].flatten().mean()
print(probabilities,focus_avg)