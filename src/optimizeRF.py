import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500]
}

# Load the data
df = pd.read_csv('out_schaffer.csv')

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

# Create a base model
rf = RandomForestClassifier(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print("Best parameters: ", best_params)