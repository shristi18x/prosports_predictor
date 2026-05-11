import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
scenario = pd.DataFrame({
    'batting_team': ['Royal Challengers Bengaluru'],
    'bowling_team': ['Mumbai Indians'],
    'city': ['Raipur'], 
    'runs_left': [9],
    'balls_left': [3],
    'wickets_left': [2],
    'target_score': [167],
    'striker_rating': [25],
    'bowler_rating': [35]
})
# index [0][1] is the probability for the Batting Team (RCB)
# index [0][0] is the probability for the Bowling Team (MI)
probabilities = pipe.predict_proba(scenario)
rcb_win = probabilities[0][1] * 100
mi_win = probabilities[0][0] * 100

#output
print("LIVE MATCH PREDICTION")
print(f"Target: 167 | Need: 9 runs from 3 balls | Wickets Left: 2")
print("-" * 30)
print(f"RCB Win Probability: {rcb_win:.2f}%")
print(f"MI Win Probability: {mi_win:.2f}%")
print("-" * 30)

if rcb_win > mi_win:
    print("Prediction: RCB is likely to pull off the win!")
else:
    print("Prediction: Mumbai Indians are the favorites to defend this.")
