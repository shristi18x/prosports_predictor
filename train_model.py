import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

#Load the data
df = pd.read_csv('cleaned_data.csv')

#Split into X and y
X = df.drop(columns=['result'])
y = df['result']

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city']),
    ('scaling', StandardScaler(), ['runs_left', 'balls_left', 'wickets_left', 'target_score', 'striker_rating', 'bowler_rating'])
], remainder='passthrough')

# 5. The Advanced Pipeline (Using Random Forest)
# Random Forest is much better at picking up 'non-linear' patterns 
# like "9 runs off 3 balls" where every ball changes everything.
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestClassifier(n_estimators=100, random_state=1, max_depth=15))
])

#Training
print("Training the Advanced Random Forest Model...")
pipe.fit(X_train, y_train)

#Accuracy Check
y_pred = pipe.predict(X_test)
final_acc = accuracy_score(y_test, y_pred) * 100
print(f"--- TRAINING COMPLETE ---")
print(f"Upgraded Model Accuracy: {final_acc:.2f}%")

pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("High-performance 'pipe.pkl' saved!")
