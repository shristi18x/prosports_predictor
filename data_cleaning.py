import pandas as pd
#load data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
#get team names
teams_per_match = deliveries.groupby('id')[['batting_team', 'bowling_team']].first().reset_index()
#target score from 1st inning
total_score_df = deliveries.groupby(['id', 'batting_team'])['total_run'].sum().reset_index()
first_innings_score = total_score_df.groupby('id').first().reset_index()
first_innings_score.rename(columns={'total_run': 'target_score'}, inplace=True)
#merge matches info, team names, and target score
match_df = matches.merge(teams_per_match, on='id')
match_df = match_df.merge(first_innings_score[['id', 'target_score']], on='id')
#current ipl teams
teams = [
    'Rajasthan Royals', 'Royal Challengers Bengaluru', 'Sunrisers Hyderabad', 
    'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 
    'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians'
]
match_df = match_df[match_df['batting_team'].isin(teams)]
match_df = match_df[match_df['bowling_team'].isin(teams)]
final_match_df = match_df[['id', 'city', 'winner', 'batting_team', 'bowling_team', 'target_score']]
print("Match Data is Cleaned!")
print(final_match_df.head())
#calculate target score
delivery_df = matches[['id', 'city', 'winner']].merge(deliveries, on='id')
delivery_df = delivery_df.merge(first_innings_score[['id', 'target_score']], on='id')
delivery_df['target_score'] = delivery_df['target_score'] + 1
#1st vs 2nd innings
teams_by_inning = delivery_df.groupby('id')['batting_team'].first().reset_index()
teams_by_inning.rename(columns={'batting_team': 'team1'}, inplace=True)
delivery_df = delivery_df.merge(teams_by_inning, on='id')
delivery_df = delivery_df[delivery_df['batting_team'] != delivery_df['team1']]
#live stats
delivery_df['current_score'] = delivery_df.groupby('id')['total_run'].cumsum()
delivery_df['runs_left'] = delivery_df['target_score'] - delivery_df['current_score']
#ball tracking
delivery_df['ball_no'] = delivery_df.groupby('id').cumcount() + 1
delivery_df['balls_left'] = 120 - delivery_df['ball_no']
delivery_df.loc[delivery_df['balls_left'] < 0, 'balls_left'] = 0
print("Chasing data created!")
print(delivery_df[['id', 'batting_team', 'runs_left', 'balls_left']].head())
#calculating wickets
def calculate_wickets(group):
    #track players seen in match
    seen_players = set()
    wickets_count = []
    current_wickets = 0
    for _, row in group.iterrows():
        s = row['batter_name']
        ns = row['non_striker_name']  
        if not seen_players:
            seen_players.add(s)
            seen_players.add(ns)
        else:
            if s not in seen_players:
                current_wickets += 1
                seen_players.add(s)
            if ns not in seen_players:
                current_wickets += 1
                seen_players.add(ns)
        if current_wickets > 10: current_wickets = 10
        wickets_count.append(current_wickets)    
    group['wickets_fallen'] = wickets_count
    return group    
delivery_df = delivery_df.groupby('id', group_keys=False).apply(calculate_wickets)
#calculate wickets left
delivery_df['wickets_left'] = 10 - delivery_df['wickets_fallen']
print(delivery_df[['id', 'batter_name', 'non_striker_name', 'wickets_left']].head(50))
#load players
players_df = pd.read_csv('players.csv')
cols_to_fix = ['Batting_Average', 'Batting_Strike_Rate', 'Economy_Rate', 'Bowling_Strike_Rate']
for col in cols_to_fix:
    players_df[col] = pd.to_numeric(players_df[col], errors='coerce')
players_df.fillna(0, inplace=True)
#avg stats across all years
player_stats = players_df.groupby('Player_Name').agg({
    'Batting_Average': 'mean',
    'Batting_Strike_Rate': 'mean',
    'Economy_Rate': 'mean',
    'Bowling_Strike_Rate': 'mean'
}).reset_index()
#create ratings
player_stats['bat_power'] = (player_stats['Batting_Average'] * player_stats['Batting_Strike_Rate']) / 100
player_stats['bowl_power'] = 10 / (player_stats['Economy_Rate'].replace(0, 10)) 
bat_map = dict(zip(player_stats['Player_Name'], player_stats['bat_power']))
bowl_map = dict(zip(player_stats['Player_Name'], player_stats['bowl_power']))
delivery_df['striker_rating'] = delivery_df['batter_name'].map(bat_map).fillna(player_stats['bat_power'].mean())
delivery_df['bowler_rating'] = delivery_df['bowler_name'].map(bowl_map).fillna(player_stats['bowl_power'].mean())
#output
def determine_result(row):
    return 1 if row['batting_team'] == row['winner'] else 0
delivery_df['result'] = delivery_df.apply(determine_result, axis=1)
final_df = delivery_df[[
    'batting_team', 'bowling_team', 'city', 'runs_left', 
    'balls_left', 'wickets_left', 'target_score', 
    'striker_rating', 'bowler_rating', 'result'
]]
Drop any rows with missing data and shuffle
final_df = final_df.dropna()
final_df = final_df.sample(final_df.shape[0])
#save in new file
final_df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved as 'cleaned_data.csv'")
print(final_df.head())
