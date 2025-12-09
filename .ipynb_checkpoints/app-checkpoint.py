import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pyspark.sql.functions import col, lower, desc
from utils import load_all_data

# --- CONFIGURATION ---
st.set_page_config(page_title="FIFA Ultimate Hub", page_icon="⚽", layout="wide")
st.title("FIFA Ultimate Hub")

# --- UTILS ---
def format_currency_custom(val):
    if pd.isna(val) or val == 0: return "€ 0"
    if val >= 1_000_000:
        return f"€ {val/1_000_000:.1f}M"
    return f"€ {val:,.0f}".replace(",", ".")

# --- 1. DATA LOADING ---
@st.cache_resource(show_spinner="Loading Datasets...")
def get_datasets_separate():
    df_p, df_t, df_c = load_all_data()
    if df_p is None: return None, None, None
    return df_p.cache(), df_t.cache(), df_c.cache()

df_players, df_teams, df_coaches = get_datasets_separate()
if df_players is None: st.stop()

# --- 2. LOOKUP TABLES ---
@st.cache_data(show_spinner=False)
def get_lookup_tables(_df_teams, _df_coaches):
    # Coach Map
    c_pdf = _df_coaches.select(col("coach_id"), col("short_name").alias("coach_name")).toPandas()
    coach_map = dict(zip(c_pdf.coach_id, c_pdf.coach_name))
    
    # Team Map
    team_col = "club_name" if "club_name" in _df_teams.columns else "team_name"
    t_pdf = _df_teams.select(
        col(team_col).alias("club_name"), 
        "coach_id", 
        col("league_name").alias("league_ref"), 
        col("overall").alias("team_rating")
    ).toPandas()
    
    team_to_coach_id = dict(zip(t_pdf.club_name, t_pdf.coach_id))
    team_to_league = dict(zip(t_pdf.club_name, t_pdf.league_ref))
    team_to_rating = dict(zip(t_pdf.club_name, t_pdf.team_rating))
    
    return coach_map, team_to_coach_id, team_to_league, team_to_rating

with st.spinner("Indexing..."):
    coach_map, team_to_coach_id, team_to_league, team_to_rating = get_lookup_tables(df_teams, df_coaches)

all_leagues = sorted(list(set([l for l in team_to_league.values() if l])))
all_works = [r[0] for r in df_players.select("work_rate").distinct().dropna().collect()] if "work_rate" in df_players.columns else []

if "fifa_version" in df_players.columns:
    all_versions = [r[0] for r in df_players.select("fifa_version").distinct().sort(desc("fifa_version")).collect()]
else: all_versions = []


# ==========================================
# MODE SELECTOR
# ==========================================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode:", ["Advanced Scouting", "Player Comparison"])
st.sidebar.divider()

# ==========================================
# MODE 1: ADVANCED SCOUTING
# ==========================================
if mode == "Advanced Scouting":
    st.sidebar.header("Filters")

    with st.sidebar.form("scout_form"):
        search_query = st.text_input("Search Player/Club")
        
        c1, c2 = st.columns(2)
        sel_league = c1.selectbox("League", ["All"] + all_leagues)
        
        if all_versions: 
            sel_ver = c2.selectbox("FIFA Version", ["All"] + all_versions)
        else: 
            sel_ver = "All"

        with st.expander("Teams & Contracts", expanded=True):
            if "club_contract_valid_until_year" in df_players.columns:
                max_contract = st.slider("Contract Expiring By", 2023, 2032, 2032)
            else: max_contract = None
            min_team_rating = st.slider("Min Team Rating", 50, 99, 50)

        with st.expander("⚽ Role & Technique", expanded=False):
            sel_role = st.text_input("Position (e.g. ST)", "").upper()
            sel_foot = st.selectbox("Preferred Foot", ["All", "Right", "Left"])
            sel_work = st.selectbox("Work Rate", ["All"] + all_works)
            min_skill = st.slider("Skill Moves ⭐", 1, 5, 1)
            min_wf = st.slider("Weak Foot ⭐", 1, 5, 1)

        with st.expander("📊 Stats & Value", expanded=False):
            val_range = st.slider("Max Value (€)", 0, 150000000, 150000000, step=500000)
            age_range = st.slider("Age", 15, 45, (16, 40))
            min_pot = st.slider("Min Potential", 40, 99, 50)
            min_pace = st.slider("Pace", 0, 99, 0)
            min_shoot = st.slider("Shooting", 0, 99, 0)
            min_phys = st.slider("Physicality", 0, 99, 0)

        submitted = st.form_submit_button("🚀 SEARCH NOW", type="primary")

    # --- SCOUTING LOGIC ---
    if submitted or 'scout_run' not in st.session_state:
        st.session_state['scout_run'] = True
        
        # 1. Lookup Clubs
        valid_clubs = None
        if sel_league != "All":
            valid_clubs = set([c for c, l in team_to_league.items() if l == sel_league])
        if min_team_rating > 50:
            strong_clubs = set([c for c, r in team_to_rating.items() if r >= min_team_rating])
            valid_clubs = valid_clubs.intersection(strong_clubs) if valid_clubs else strong_clubs

        # 2. Spark Filter
        filtered = df_players
        if valid_clubs is not None:
            if not valid_clubs: filtered = filtered.filter("1=0")
            else: filtered = filtered.filter(col("club_name").isin(list(valid_clubs)))

        if sel_ver != "All": filtered = filtered.filter(col("fifa_version") == sel_ver)
        if search_query:
            t = search_query.lower()
            filtered = filtered.filter((lower(col("short_name")).contains(t)) | (lower(col("club_name")).contains(t)))

        filtered = filtered.filter(
            (col("value_eur") <= val_range) & (col("age").between(age_range[0], age_range[1])) &
            (col("potential") >= min_pot) & (col("pace") >= min_pace) &
            (col("shooting") >= min_shoot) & (col("physic") >= min_phys)
        )
        if max_contract: filtered = filtered.filter(col("club_contract_valid_until_year") <= max_contract)
        
        if sel_role: filtered = filtered.filter(col("player_positions").contains(sel_role))
        if sel_foot != "All": filtered = filtered.filter(col("preferred_foot") == sel_foot)
        if sel_work != "All": filtered = filtered.filter(col("work_rate") == sel_work)
        if min_skill > 1: filtered = filtered.filter(col("skill_moves") >= min_skill)
        if min_wf > 1: filtered = filtered.filter(col("weak_foot") >= min_wf)

        # Select Columns
        cols_fetch = ["short_name", "fifa_version", "age", "overall", "potential", "club_name", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
        # Intersection to avoid errors
        cols_fetch = [c for c in cols_fetch if c in df_players.columns]

        pdf = filtered.select(cols_fetch).orderBy(desc("overall"), desc("fifa_version")).limit(200).toPandas()
        st.session_state['res_scout'] = pdf

    # --- SCOUTING VISUALIZATION ---
    if 'res_scout' in st.session_state and not st.session_state['res_scout'].empty:
        pdf_view = st.session_state['res_scout'].copy()
        
        # Enrichment
        pdf_view["League"] = pdf_view["club_name"].map(team_to_league)
        pdf_view["Coach ID"] = pdf_view["club_name"].map(team_to_coach_id)
        pdf_view["Coach"] = pdf_view["Coach ID"].map(coach_map)
        
        if "fifa_version" in pdf_view.columns:
            pdf_view["Edition"] = pdf_view["fifa_version"].apply(lambda x: f"FIFA {int(x)}" if pd.notnull(x) else "")
        if "value_eur" in pdf_view.columns:
            pdf_view["Value"] = pdf_view["value_eur"].apply(format_currency_custom)

        # Final Columns
        cols_final = ["short_name", "Edition", "age", "overall", "potential", "club_name", "League", "Coach", "Value"]
        cols_final = [c for c in cols_final if c in pdf_view.columns]

        st.subheader(f"Search Results: {len(pdf_view)}")
        st.dataframe(
            pdf_view[cols_final], 
            use_container_width=True, 
            height=700, 
            hide_index=True,
            column_config={
                "short_name": "Name",
                "club_name": "Club",
                "age": "Age",
                "overall": st.column_config.ProgressColumn("Overall", format="%d", min_value=0, max_value=100)
            }
        )
    
    elif submitted:
        st.warning("No players found matching your criteria.")
    else:
        st.info("Set filters on the left and click SEARCH NOW.")


# ==========================================
# MODE 2: PLAYER COMPARISON
# ==========================================
elif mode == "Player Comparison":
    st.markdown("## ⚖️ Direct Comparison")
    st.markdown("Search and select up to 3 players to compare side-by-side.")

    # 1. Search Bar to find players
    col_search, col_ver = st.columns([3, 1])
    search_cmp = col_search.text_input("Search player name to add:", placeholder="e.g. Messi")
    
    if all_versions:
        ver_cmp = col_ver.selectbox("DB Version", all_versions, index=0)
    else: ver_cmp = None

    # Instant Search
    candidates = pd.DataFrame()
    if search_cmp:
        filt = df_players.filter(lower(col("short_name")).contains(search_cmp.lower()))
        if ver_cmp: filt = filt.filter(col("fifa_version") == ver_cmp)
        
        # Get top 10 candidates
        candidates = filt.select("short_name", "club_name", "overall", "player_positions").limit(10).toPandas()
    
    # 2. Multichoice Selector
    if not candidates.empty:
        candidates["label"] = candidates["short_name"] + " (" + candidates["club_name"] + ") - " + candidates["overall"].astype(str)
        options = candidates["label"].tolist()
    else:
        options = []

    # Use session state to remember selection
    if 'selected_players' not in st.session_state: st.session_state['selected_players'] = []

    # Dropdown to add players
    new_selection = st.selectbox("Select from results:", [""] + options)
    
    # Add to list if selected and unique
    if new_selection and new_selection != "":
        if new_selection not in st.session_state['selected_players']:
            if len(st.session_state['selected_players']) < 3:
                st.session_state['selected_players'].append(new_selection)
            else:
                st.warning("Maximum 3 players for comparison.")

    # Reset Button
    if st.button("🗑️ Clear Comparison"):
        st.session_state['selected_players'] = []
        st.rerun()

    # 3. Comparison Visualization
    selected_labels = st.session_state['selected_players']
    
    if selected_labels:
        # Extract name logic
        names = [s.split(" (")[0] for s in selected_labels]
        
        # Spark Query
        stats_cols = ["short_name", "overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "value_eur", "age", "height_cm", "weight_kg", "weak_foot", "skill_moves"]
        stats_cols = [c for c in stats_cols if c in df_players.columns]
        
        cmp_df = df_players.filter(col("short_name").isin(names))
        if ver_cmp: cmp_df = cmp_df.filter(col("fifa_version") == ver_cmp)
        
        # Convert to Pandas
        pdf_cmp = cmp_df.select(stats_cols).dropDuplicates(["short_name"]).toPandas()

        if not pdf_cmp.empty:
            st.divider()
            
            # --- DATA TABLE ---
            st.subheader("📋 Technical Sheet")
            # Transpose: Cols = Players, Rows = Stats
            pdf_cmp = pdf_cmp.set_index("short_name")
            st.dataframe(pdf_cmp.T, use_container_width=True)

            # --- RADAR CHART (SPIDER PLOT) ---
            st.subheader("🕸️ Radar Chart")
            
            categories = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            categories = [c for c in categories if c in pdf_cmp.columns]

            fig = go.Figure()

            for player_name, row in pdf_cmp.iterrows():
                values = [row[c] for c in categories]
                values += [values[0]] # Close the circle
                cats_closed = categories + [categories[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=cats_closed,
                    fill='toself',
                    name=player_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True
            )
            
            

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Error retrieving data for selected players.")