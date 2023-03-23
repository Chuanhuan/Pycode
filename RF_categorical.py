
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
# data = pd.read_csv('data.csv')

data = pd.DataFrame(np.random.randint(2, size=(10, 5)))
X_test = pd.DataFrame(np.random.randint(2, size=(10, 4)))


# Split data into X and y
X = data.drop(data.columns[0], axis=1)
y = data[data.columns[0]]
print(type( data.columns ))
# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Create a random forest classifier object
rfc = RandomForestClassifier()

# Train the model using the training sets
rfc.fit(X, y)

# Predict the response for test dataset
y_pred = rfc.predict(X_test)

print(y_pred)
