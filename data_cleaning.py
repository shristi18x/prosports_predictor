import pandas as pd

# 1. Load the data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

# 2. Get the Team Names from deliveries
# Since matches.csv is missing team names, we find the unique teams for each match 'id'
teams_per_match = deliveries.groupby('id')[['batting_team', 'bowling_team']].first().reset_index()

# 3. Calculate the Total Runs (Target Score) from the 1st Innings
# We group by match 'id' and 'batting_team' to get the total runs scored by each team
total_score_df = deliveries.groupby(['id', 'batting_team'])['total_run'].sum().reset_index()

# In IPL, the first team to bat sets the target. We take the first team that appears for each ID.
first_innings_score = total_score_df.groupby('id').first().reset_index()
first_innings_score.rename(columns={'total_run': 'target_score'}, inplace=True)

# 4. Merge everything into one clean Match DataFrame
# We combine the matches info, the team names, and the target score
match_df = matches.merge(teams_per_match, on='id')
match_df = match_df.merge(first_innings_score[['id', 'target_score']], on='id')

# 5. Filter for Current IPL Teams (2026 Ready)
# This keeps the model accurate and avoids "dead" teams like Pune Warriors
teams = [
    'Rajasthan Royals', 'Royal Challengers Bengaluru', 'Sunrisers Hyderabad', 
    'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 
    'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians'
]

# We filter both batting and bowling columns
match_df = match_df[match_df['batting_team'].isin(teams)]
match_df = match_df[match_df['bowling_team'].isin(teams)]

# 6. Final Clean: Keep only the columns we need for AI
final_match_df = match_df[['id', 'city', 'winner', 'batting_team', 'bowling_team', 'target_score']]

print("Step 3 Complete: Match Data is Cleaned!")
print(final_match_df.head())

# --- STEP 4: BUILDING THE CHASE LOGIC (Refined) ---

# 1. Merge carefully. We only take 'id', 'city', 'winner', and 'target_score' from matches
# and let 'deliveries' provide the 'batting_team' and 'bowling_team'
delivery_df = matches[['id', 'city', 'winner']].merge(deliveries, on='id')

# 2. Add 1 to target_score calculation (using the first innings total)
# We need to make sure 'target_score' is attached to each ball
delivery_df = delivery_df.merge(first_innings_score[['id', 'target_score']], on='id')
delivery_df['target_score'] = delivery_df['target_score'] + 1

# 3. Identify 1st vs 2nd Innings
# We find the team that batted first in each match
teams_by_inning = delivery_df.groupby('id')['batting_team'].first().reset_index()
teams_by_inning.rename(columns={'batting_team': 'team1'}, inplace=True)

delivery_df = delivery_df.merge(teams_by_inning, on='id')

# Keep only 2nd innings (where current batting team is NOT team1)
delivery_df = delivery_df[delivery_df['batting_team'] != delivery_df['team1']]

# 4. Calculate Live Stats
delivery_df['current_score'] = delivery_df.groupby('id')['total_run'].cumsum()
delivery_df['runs_left'] = delivery_df['target_score'] - delivery_df['current_score']

# Ball tracking
delivery_df['ball_no'] = delivery_df.groupby('id').cumcount() + 1
delivery_df['balls_left'] = 120 - delivery_df['ball_no']
delivery_df.loc[delivery_df['balls_left'] < 0, 'balls_left'] = 0

# 5. NEW: Calculate Wickets Left
# We need to see how many wickets have fallen at each ball
# Your deliveries file usually has a way to identify a wicket. 
# Let's check for 'player_dismissed' or similar in the next step, 
# but for now, let's just see if this runs!

print("STEP 4 SUCCESS: Chasing data created!")
print(delivery_df[['id', 'batting_team', 'runs_left', 'balls_left']].head())

# --- STEP 5: INFERRING WICKETS VIA "NEW PLAYER" LOGIC ---

def calculate_wickets(group):
    # Track players seen in this specific match
    seen_players = set()
    wickets_count = []
    current_wickets = 0
    
    for _, row in group.iterrows():
        s = row['batter_name']
        ns = row['non_striker_name']
        
        # If this is the very first ball, initialize the seen_players
        if not seen_players:
            seen_players.add(s)
            seen_players.add(ns)
        else:
            # If a batter appears who we haven't seen before in this match...
            if s not in seen_players:
                current_wickets += 1
                seen_players.add(s)
            if ns not in seen_players:
                current_wickets += 1
                seen_players.add(ns)
                
        # Cap it at 10 wickets max
        if current_wickets > 10: current_wickets = 10
        wickets_count.append(current_wickets)
        
    group['wickets_fallen'] = wickets_count
    return group

# Apply this logic to every match ID
delivery_df = delivery_df.groupby('id', group_keys=False).apply(calculate_wickets)

# Calculate Wickets Left
delivery_df['wickets_left'] = 10 - delivery_df['wickets_fallen']

print("INFERRED WICKET LOGIC COMPLETE!")
print(delivery_df[['id', 'batter_name', 'non_striker_name', 'wickets_left']].head(50))

# --- STEP 6: PLAYER STRENGTH INDEX (SAFE VERSION) ---

# 1. Load players
players_df = pd.read_csv('players.csv')

# 2. CLEANING: Force columns to be numeric and handle the "20.5.18" style errors
cols_to_fix = ['Batting_Average', 'Batting_Strike_Rate', 'Economy_Rate', 'Bowling_Strike_Rate']

for col in cols_to_fix:
    # 'coerce' turns errors into NaN (empty)
    players_df[col] = pd.to_numeric(players_df[col], errors='coerce')

# Fill those newly created NaNs with 0 so the math doesn't break
players_df.fillna(0, inplace=True)

# 3. Create Career Profiles (Average stats across all years)
player_stats = players_df.groupby('Player_Name').agg({
    'Batting_Average': 'mean',
    'Batting_Strike_Rate': 'mean',
    'Economy_Rate': 'mean',
    'Bowling_Strike_Rate': 'mean'
}).reset_index()

# 4. Create the "Power Ratings"
player_stats['bat_power'] = (player_stats['Batting_Average'] * player_stats['Batting_Strike_Rate']) / 100
# For bowlers, lower economy is better (replace 0 with 10 to avoid division by zero)
player_stats['bowl_power'] = 10 / (player_stats['Economy_Rate'].replace(0, 10)) 

# 5. Map these ratings to our delivery_df
bat_map = dict(zip(player_stats['Player_Name'], player_stats['bat_power']))
bowl_map = dict(zip(player_stats['Player_Name'], player_stats['bowl_power']))

# Use the mean of the ratings for any player not found in the CSV
delivery_df['striker_rating'] = delivery_df['batter_name'].map(bat_map).fillna(player_stats['bat_power'].mean())
delivery_df['bowler_rating'] = delivery_df['bowler_name'].map(bowl_map).fillna(player_stats['bowl_power'].mean())
# --- STEP 7: FINAL EXPORT ---

# 1. Create the 'result' column
def determine_result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(determine_result, axis=1)

# 2. Extract only the columns the AI needs (Including new ratings!)
final_df = delivery_df[[
    'batting_team', 'bowling_team', 'city', 'runs_left', 
    'balls_left', 'wickets_left', 'target_score', 
    'striker_rating', 'bowler_rating', 'result'
]]

# 3. Final Polish: Drop any rows with missing data and shuffle
final_df = final_df.dropna()
final_df = final_df.sample(final_df.shape[0])

# 4. Save this to a NEW CSV file
final_df.to_csv('cleaned_data.csv', index=False)

print("--- ALL STEPS COMPLETE WITH PLAYER STATS ---")
print("Cleaned data saved as 'cleaned_data.csv'")
print(final_df.head())