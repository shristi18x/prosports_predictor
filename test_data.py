import pandas as pd

# Load the files
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
players= pd.read_csv('players.csv')

print("--- MATCHES.CSV COLUMNS ---")
print(matches.columns.tolist())

print("\n--- DELIVERIES.CSV COLUMNS ---")
print(deliveries.columns.tolist())

print("\n--- PLAYERS.CSV COLUMNS ---")
print(players.columns.tolist())

print("\n--- DATA PREVIEW ---")
print("Matches Head:")
print(matches.head(1)) 

h2h = deliveries.groupby(['batter_name', 'bowler_name']).agg({
    'batsman_run': 'sum',
    'id': 'count' # This counts every row (every ball)
}).reset_index()

# 3. Rename columns for clarity
h2h.columns = ['batter_name', 'bowler_name', 'h2h_runs', 'h2h_balls']

# 4. Calculate H2H Strike Rate
h2h['h2h_strike_rate'] = (h2h['h2h_runs'] / h2h['h2h_balls']) * 100

# 5. Save this to use in your App
h2h.to_csv('matchups.csv', index=False)

print("matchups.csv generated successfully!")