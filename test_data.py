import pandas as pd

# Load the files
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
players= pd.read_csv('players.csv')

print("MATCHES.CSV COLUMNS")
print(matches.columns.tolist())

print("\nDELIVERIES.CSV COLUMNS")
print(deliveries.columns.tolist())

print("\nPLAYERS.CSV COLUMNS")
print(players.columns.tolist())

print("\nDATA PREVIEW")
print("Matches Head:")
print(matches.head(1)) 

h2h = deliveries.groupby(['batter_name', 'bowler_name']).agg({
    'batsman_run': 'sum',
    'id': 'count' )
}).reset_index()

h2h.columns = ['batter_name', 'bowler_name', 'h2h_runs', 'h2h_balls']

#Calculate H2H Strike Rate
h2h['h2h_strike_rate'] = (h2h['h2h_runs'] / h2h['h2h_balls']) * 100

# 5. Save this to use in your App
h2h.to_csv('matchups.csv', index=False)

print("matchups.csv generated successfully!")
