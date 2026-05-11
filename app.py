import streamlit as st
import pickle
import pandas as pd

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="IPL Match Predictor 2026", layout="wide")

# 2. CUSTOM CSS FOR PRO LOOK
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e445e;
    }
    .win-text {
        color: #00ffcc;
        font-size: 40px;
        font-weight: bold;
    }
    .loss-text {
        color: #ff4b4b;
        font-size: 40px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SAFE DATA LOADING (LOAD PLAYERS WITH CORRECT COLUMNS)
@st.cache_data
def load_player_data():
    try:
        player_df = pd.read_csv('players.csv')
        
        # Mapping your specific CSV columns to dictionaries
        # Batting_Strike_Rate for strikers, Bowling_Strike_Rate for bowlers
        batting_dict = dict(zip(player_df['Player_Name'], player_df['Batting_Strike_Rate']))
        bowling_dict = dict(zip(player_df['Player_Name'], player_df['Bowling_Strike_Rate']))
        
        return batting_dict, bowling_dict
    except Exception as e:
        st.error(f"Error loading players.csv: {e}")
        # Fallback in case of error
        return {"Default Player": 100.0}, {"Default Player": 50.0}
    
@st.cache_data
def load_matchup_data():
    try:
        return pd.read_csv('matchups.csv')
    except:
        return None

matchup_df = load_matchup_data()

# Load the dictionaries
batting_stats, bowling_stats = load_player_data()
player_names = sorted(list(batting_stats.keys()))

# 4. LOAD MODEL
pipe = pickle.load(open('pipe.pkl', 'rb'))

# 5. DATA CONSTANTS
teams = ['Rajasthan Royals', 'Royal Challengers Bengaluru', 'Sunrisers Hyderabad', 
         'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 
         'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Dubai', 'Navi Mumbai', 'Lucknow', 'Guwahati',
       'Mohali']

# --- SIDEBAR FOR SETTINGS ---
st.sidebar.header("Match Settings")
batting_team = st.sidebar.selectbox('Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Bowling Team', sorted(teams), index=1)
selected_city = st.sidebar.selectbox('Host City', sorted(cities))
target = st.sidebar.number_input('Target Score', min_value=1, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("Player Matchup")
# Selecting the specific batter and bowler from your CSV list
selected_striker = st.sidebar.selectbox("Current Batter", player_names)
selected_bowler = st.sidebar.selectbox("Current Bowler", player_names)

# Assigning the correct numerical ratings from your CSV columns
striker_rating = batting_stats[selected_striker]
bowler_rating = bowling_stats[selected_bowler]

# --- MAIN PAGE LAYOUT ---
st.title("IPL Match Win Predictor Dashboard")
st.markdown("---")

# Input columns
col1, col2, col3 = st.columns(3)
with col1:
    score = st.number_input('Current Score', min_value=0, value=100)
with col2:
    overs = st.number_input('Overs Completed (0-20)', min_value=0.0, max_value=20.0, value=15.0, step=0.1)
with col3:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, value=3)

# --- FIXED CALCULATIONS ---
completed_overs = int(overs) 
extra_balls = int((overs * 10) % 10) 

# Ball count validation
if extra_balls > 5:
    st.error("Invalid input: An over cannot have more than 5 balls (e.g., 19.5 is the limit before 20.0)")

total_balls_bowled = (completed_overs * 6) + extra_balls
balls_left = 120 - total_balls_bowled
runs_left = target - score
wickets_left = 10 - wickets

# Default to the general ratings from players.csv
final_striker_sr = striker_rating 

if matchup_df is not None:
    # Filter for the specific batter vs bowler
    stats = matchup_df[(matchup_df['batter_name'] == selected_striker) & 
                       (matchup_df['bowler_name'] == selected_bowler)]
    
    if not stats.empty:
        # If they have faced at least 6 balls, use the H2H Strike Rate
        if stats['h2h_balls'].values[0] >= 6:
            final_striker_sr = stats['h2h_strike_rate'].values[0]
            st.sidebar.info(f"H2H Stats Found! SR: {round(final_striker_sr, 1)}")

# --- PREDICTION LOGIC ---
if st.button('ANALYZE WIN PROBABILITY', use_container_width=True):
    
    # Initialize variables
    win = 0.0
    loss = 0.0

    # 1. First, check if the game is already over
    if balls_left <= 0:
        if runs_left > 0:
            win, loss = 0.0, 1.0
            st.error(f"Match Over! {bowling_team} won.")
        else:
            win, loss = 1.0, 0.0
            st.success(f"Match Over! {batting_team} won.")
    
    # 2. Check for "Impossible" Scenarios (The Kill Switch)
    elif runs_left > (balls_left * 6):
        win = 0.0
        loss = 1.0
        st.warning("Mathematically impossible for the batting team to win without no balls or multiple wides!")
        
    elif runs_left <= 0:
        win = 1.0
        loss = 0.0
        st.success(f"{batting_team} has already chased the target!")
        
    else:
        
        # 3. If "Possible," ask the AI Model using CSV ratings
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 
            'bowling_team': [bowling_team], 
            'city': [selected_city],
            'runs_left': [runs_left], 
            'balls_left': [balls_left], 
            'wickets_left': [wickets_left],
            'target_score': [target], 
            'striker_rating': [final_striker_sr], 
            'bowler_rating': [bowler_rating]
        })
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

    # --- DISPLAY RESULTS ---
    st.markdown("### Match Equation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Runs Needed", runs_left)
    m2.metric("Balls Remaining", balls_left)
    m3.metric("Wickets in Hand", wickets_left)

    st.markdown("---")
    st.subheader("Win Probability")
    
    st.progress(win)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<p class='win-text'>{batting_team}: {round(win*100)}%</p>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p class='loss-text'>{bowling_team}: {round(loss*100)}%</p>", unsafe_allow_html=True)