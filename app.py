import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pyspark.sql.functions import col, lower, desc
from utils import load_all_data
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="FIFA Scout", layout="wide")
st.title("FIFA Scout")

# --- UTILS ---
def format_currency_custom(val):
    if pd.isna(val) or val == 0: return "€ 0"
    if val >= 1_000_000:
        return f"€ {val/1_000_000:.1f}M"
    return f"€ {val:,.0f}".replace(",", ".")

# --- DATA LOADING ---
@st.cache_resource(show_spinner="Loading Datasets...")
def get_datasets_separate():
    df_p, df_t, df_c = load_all_data()
    if df_p is None: return None, None, None
    
    if "coach_id" in df_t.columns: df_t = df_t.withColumn("coach_id", col("coach_id").cast("int"))
    if "coach_id" in df_c.columns: df_c = df_c.withColumn("coach_id", col("coach_id").cast("int"))
    if "fifa_version" in df_t.columns: df_t = df_t.withColumn("fifa_version", col("fifa_version").cast("int"))
    
    return df_p.cache(), df_t.cache(), df_c.cache()

df_players, df_teams, df_coaches = get_datasets_separate()
if df_players is None: st.stop()

# --- 2. LOOKUP TABLES ---
@st.cache_data(show_spinner=False)
def get_lookups(_df_teams, _df_coaches):
    # Map Allenatori semplice: ID -> Nome
    c_pdf = _df_coaches.select("coach_id", col("short_name").alias("coach_name")).distinct().toPandas()
    coach_map = dict(zip(c_pdf.coach_id, c_pdf.coach_name))
    
    t_col = "club_name" if "club_name" in _df_teams.columns else "team_name"
    t_pdf = _df_teams.select(col(t_col).alias("club_name"), col("league_name").alias("league_ref")).distinct().toPandas()
    team_to_league = dict(zip(t_pdf.club_name, t_pdf.league_ref))
    
    t_c_pdf = _df_teams.select(col(t_col).alias("club_name"), "coach_id").distinct().toPandas()
    team_to_coach_id = dict(zip(t_c_pdf.club_name, t_c_pdf.coach_id))
    
    return coach_map, team_to_league, team_to_coach_id

with st.spinner("Indexing..."):
    coach_map, team_to_league, team_to_coach_id = get_lookups(df_teams, df_coaches)

all_leagues = sorted(list(set([l for l in team_to_league.values() if l])))
all_works = [r[0] for r in df_players.select("work_rate").distinct().dropna().collect()] if "work_rate" in df_players.columns else []

if "fifa_version" in df_players.columns:
    all_versions = [int(r[0]) for r in df_players.select("fifa_version").distinct().sort(desc("fifa_version")).collect()]
else: all_versions = []

def get_best_lineup(df):
    df = df.copy()
    
    def is_goalkeeper(pos):
        if not pos: return False
        return 'GK' in pos.split(',')[0]

    df['is_gk'] = df['player_positions'].apply(is_goalkeeper)
    
    df = df.sort_values('overall', ascending=False)
    
    # POOL PORTIERI e POOL MOVIMENTO
    gk_pool = df[df['is_gk'] == True]
    outfield_pool = df[df['is_gk'] == False]
    
    if not gk_pool.empty:
        starter_gk = gk_pool.head(1)
    else:
        starter_gk = pd.DataFrame() 
        
    starter_outfield = outfield_pool.head(10)
    
    # Uniamo
    starters = pd.concat([starter_gk, starter_outfield])
    
    # PANCHINA
    bench = df[~df['short_name'].isin(starters['short_name'])]
    
    return starters, bench


def create_pitch_plot(players_df):
    fig = go.Figure()

    field_shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, layer="below", line=dict(width=0), fillcolor="#43a047"),
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, layer="below", line=dict(color="white", width=2)),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, layer="below", line=dict(color="white", width=2)),
        dict(type="circle", x0=40, y0=40, x1=60, y1=60, layer="below", line=dict(color="white", width=2)),
        dict(type="rect", x0=0, y0=20, x1=17, y1=80, layer="below", line=dict(color="white", width=2)),
        dict(type="rect", x0=83, y0=20, x1=100, y1=80, layer="below", line=dict(color="white", width=2)),
        dict(type="rect", x0=-2, y0=45, x1=0, y1=55, layer="below", line=dict(color="white", width=2)),
        dict(type="rect", x0=100, y0=45, x1=102, y1=55, layer="below", line=dict(color="white", width=2)),
    ]
    fig.update_layout(shapes=field_shapes)

    x_positions = {'GK': 5, 'DEF': 25, 'MID': 55, 'FWD': 85}

    def get_role_group_plot(pos):
        if not pos: return "MID"
        main = pos.split(',')[0]
        if 'GK' in main: return 'GK'
        if any(x in main for x in ['B', 'CB', 'WB', 'LB', 'RB']): return 'DEF'
        if any(x in main for x in ['M', 'CDM', 'CAM', 'CM', 'LM', 'RM']): return 'MID'
        return 'FWD'

    roles = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
    
    for _, player in players_df.iterrows():
        r = get_role_group_plot(player['player_positions'])
        roles[r].append(player)

    def get_y_coords(count):
        if count == 1: return [50]
        step = 100 / (count + 1)
        return [step * (i+1) for i in range(count)][::-1]

    for role, p_list in roles.items():
        if not p_list: continue
        ys = get_y_coords(len(p_list))
        x = x_positions[role]
        
        for i, player in enumerate(p_list):
            hover_text = f"<b>{player['short_name']}</b><br>OVR: {player['overall']}<br>Pos: {player['player_positions']}"
            
            #point
            fig.add_trace(go.Scatter(
                x=[x], y=[ys[i]],
                mode='markers+text',
                marker=dict(size=22, color='white', line=dict(color='black', width=2)),
                text=str(player['overall']),
                textposition="middle center",
                textfont=dict(color='black', size=11, family="Arial Black"),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False
            ))

            #name (below)
            fig.add_trace(go.Scatter(
                x=[x], y=[ys[i] - 8],
                mode='text',
                text=f"<b>{player['short_name']}</b>",
                textposition="bottom center",
                textfont=dict(color='white', size=11, family="Arial", shadow="2px 2px 2px black"),
                hoverinfo="skip",
                showlegend=False
            ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-5, 105]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
        plot_bgcolor="#43a047",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        dragmode=False
    )
    return fig


# ==========================================
# MODE SELECTOR
# ==========================================
st.sidebar.title("Navigation")
# RIMOSSO "Coaches Manager"
mode = st.sidebar.radio("Menu", ["Advanced Scouting", "Player Comparison", "Team Tactics"])
st.sidebar.divider()

# ==========================================
# MODE 1: ADVANCED SCOUTING
# ==========================================
if mode == "Advanced Scouting":
    st.sidebar.header("Filters")
    with st.sidebar.form("scout_form"):
        search_query = st.text_input("Search Player Name")
        c1, c2 = st.columns(2)
        sel_league = c1.selectbox("League", ["All"] + all_leagues)
        
        if sel_league == "All": available_teams = sorted(list(team_to_league.keys()))
        else: available_teams = sorted([t for t, l in team_to_league.items() if l == sel_league])   
        
        sel_team = c2.selectbox("Club", ["All"] + available_teams)
        if all_versions: sel_ver = st.selectbox("FIFA Version", ["All"] + all_versions)
        else: sel_ver = "All"
        
        with st.expander("Contracts & Ratings", expanded=True):
            if "club_contract_valid_until_year" in df_players.columns:
                max_contract = st.slider("Contract Expiring By", 2023, 2050, 2032)
            else: max_contract = None
            if sel_team == "All": min_team_rating = st.slider("Min Team Rating", 50, 99, 50)
            else: min_team_rating = 50
            
        with st.expander("Role & Technique", expanded=False):
            sel_role = st.text_input("Position (e.g. ST)", "").upper()
            sel_foot = st.selectbox("Preferred Foot", ["All", "Right", "Left"])
            sel_work = st.selectbox("Work Rate", ["All"] + all_works)
            min_skill = st.slider("Skill Moves", 1, 5, 1)
            min_wf = st.slider("Weak Foot", 1, 5, 1)
            
        with st.expander("Stats & Value", expanded=False):
            val_range = st.slider("Max Value", 0, 250000000, 250000000, step=500000)
            age_range = st.slider("Age", 15, 45, (16, 40))
            min_pot = st.slider("Min Potential", 40, 99, 50)
            min_pace = st.slider("Pace", 0, 99, 0)
            min_shoot = st.slider("Shooting", 0, 99, 0)
            min_phys = st.slider("Physicality", 0, 99, 0)
        submitted = st.form_submit_button("SEARCH NOW", type="primary")

    if submitted or 'scout_run' not in st.session_state:
        st.session_state['scout_run'] = True
        filtered = df_players
        if sel_team != "All": filtered = filtered.filter(col("club_name") == sel_team)
        else:
            valid_clubs = None
            if sel_league != "All": valid_clubs = set([c for c, l in team_to_league.items() if l == sel_league])
            if min_team_rating > 50:
                strong_clubs = set([c for c, r in team_to_rating.items() if r >= min_team_rating]) 
                pass 
                
            if valid_clubs is not None:
                if not valid_clubs: filtered = filtered.filter("1=0")
                else: filtered = filtered.filter(col("club_name").isin(list(valid_clubs)))
        if sel_ver != "All": filtered = filtered.filter(col("fifa_version") == sel_ver)
        if search_query:
            t = search_query.lower()
            filtered = filtered.filter(lower(col("short_name")).contains(t))
        filtered = filtered.filter((col("value_eur") <= val_range) & (col("age").between(age_range[0], age_range[1])) & (col("potential") >= min_pot) & (col("pace") >= min_pace) & (col("shooting") >= min_shoot) & (col("physic") >= min_phys))
        if max_contract: filtered = filtered.filter(col("club_contract_valid_until_year") <= max_contract)
        if sel_role: filtered = filtered.filter(col("player_positions").contains(sel_role))
        if sel_foot != "All": filtered = filtered.filter(col("preferred_foot") == sel_foot)
        if sel_work != "All": filtered = filtered.filter(col("work_rate") == sel_work)
        if min_skill > 1: filtered = filtered.filter(col("skill_moves") >= min_skill)
        if min_wf > 1: filtered = filtered.filter(col("weak_foot") >= min_wf)
        
        cols_fetch = ["short_name", "fifa_version", "age", "overall", "potential", "club_name", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
        cols_fetch = [c for c in cols_fetch if c in df_players.columns]
        
        pdf = filtered.select(cols_fetch).orderBy(desc("overall"), desc("fifa_version")).limit(200).toPandas()
        st.session_state['res_scout'] = pdf

    if 'res_scout' in st.session_state and not st.session_state['res_scout'].empty:
        pdf_view = st.session_state['res_scout'].copy()
        
        pdf_view["League"] = pdf_view["club_name"].map(team_to_league)
        # Recupero coach da mappa semplice
        pdf_view["Coach ID"] = pdf_view["club_name"].map(team_to_coach_id)
        pdf_view["Coach"] = pdf_view["Coach ID"].map(coach_map)

        if "fifa_version" in pdf_view.columns: pdf_view["Edition"] = pdf_view["fifa_version"].apply(lambda x: f"FIFA {int(x)}" if pd.notnull(x) else "")
        if "value_eur" in pdf_view.columns: pdf_view["Value"] = pdf_view["value_eur"].apply(format_currency_custom)
        
        display_cols = ["short_name", "Edition", "age", "overall", "potential", "club_name", "League", "Coach", "Value"]
        display_cols = [c for c in display_cols if c in pdf_view.columns]
        
        st.subheader(f"Search Results: {len(pdf_view)}")
        st.dataframe(pdf_view[display_cols], use_container_width=True, height=700, hide_index=True, column_config={"short_name": "Name", "club_name": "Club", "age": "Age", "overall": st.column_config.ProgressColumn("Overall", format="%d", min_value=0, max_value=100)})
    elif submitted: st.warning("No players found matching your criteria.")
    else: st.info("Set filters on the left and click SEARCH NOW.")

# ==========================================
# MODE 2: PLAYER COMPARISON
# ==========================================
elif mode == "Player Comparison":
    st.markdown("## Direct Comparison")
    col_search, col_ver = st.columns([3, 1])
    search_cmp = col_search.text_input("Search player name to add:", placeholder="e.g. Messi")
    if all_versions: ver_cmp = col_ver.selectbox("DB Version", all_versions, index=0)
    else: ver_cmp = None
    candidates = pd.DataFrame()
    if search_cmp:
        filt = df_players.filter(lower(col("short_name")).contains(search_cmp.lower()))
        if ver_cmp: filt = filt.filter(col("fifa_version") == ver_cmp)
        candidates = filt.select("short_name", "club_name", "overall", "player_positions").limit(10).toPandas()
    if not candidates.empty:
        candidates["label"] = candidates["short_name"] + " (" + candidates["club_name"] + ") - " + candidates["overall"].astype(str)
        options = candidates["label"].tolist()
    else: options = []
    if 'selected_players' not in st.session_state: st.session_state['selected_players'] = []
    new_selection = st.selectbox("Select from results:", [""] + options)
    if new_selection and new_selection != "":
        if new_selection not in st.session_state['selected_players']:
            if len(st.session_state['selected_players']) < 3: st.session_state['selected_players'].append(new_selection)
            else: st.warning("Maximum 3 players for comparison.")
    if st.button("Clear Comparison"):
        st.session_state['selected_players'] = []
        st.rerun()
    selected_labels = st.session_state['selected_players']
    if selected_labels:
        names = [s.split(" (")[0] for s in selected_labels]
        stats_cols = ["short_name", "overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "value_eur", "age", "height_cm", "weight_kg", "weak_foot", "skill_moves"]
        stats_cols = [c for c in stats_cols if c in df_players.columns]
        cmp_df = df_players.filter(col("short_name").isin(names))
        if ver_cmp: cmp_df = cmp_df.filter(col("fifa_version") == ver_cmp)
        pdf_cmp = cmp_df.select(stats_cols).dropDuplicates(["short_name"]).toPandas()
        if not pdf_cmp.empty:
            st.divider()
            st.subheader("Technical Sheet")
            pdf_cmp = pdf_cmp.set_index("short_name")
            st.dataframe(pdf_cmp.T, use_container_width=True)
            st.subheader("Radar Chart")
            categories = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            categories = [c for c in categories if c in pdf_cmp.columns]
            fig = go.Figure()
            for player_name, row in pdf_cmp.iterrows():
                values = [row[c] for c in categories]
                values += [values[0]]
                cats_closed = categories + [categories[0]]
                fig.add_trace(go.Scatterpolar(r=values, theta=cats_closed, fill='toself', name=player_name))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else: st.error("Error retrieving data for selected players.")

# ==========================================
# MODE 3: TEAM TACTICS
# ==========================================
elif mode == "Team Tactics":
    st.header("Team Squad Builder")
    st.markdown("Select a team and year to visualize the **Best 11** on the pitch.")
    
    col_l, col_t, col_y = st.columns(3)
    sel_league_tac = col_l.selectbox("Select League", all_leagues)
    clubs_in_league = sorted([t for t, l in team_to_league.items() if l == sel_league_tac])
    sel_club_tac = col_t.selectbox("Select Club", clubs_in_league)
    sel_year_tac = col_y.selectbox("Select Year", all_versions, index=0) if all_versions else None
    
    if st.button("Show Squad"):
        with st.spinner(f"Scouting {sel_club_tac} ({sel_year_tac})..."):
            df_squad = df_players.filter((col("club_name") == sel_club_tac) & (col("fifa_version") == sel_year_tac))
            
            
            cols_needed = ["short_name", "player_positions", "overall", "potential", "age", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
            cols_needed = [c for c in cols_needed if c in df_players.columns]
            squad_pdf = df_squad.select(cols_needed).orderBy(desc("overall")).toPandas()
            
            if not squad_pdf.empty:
                
                starters, bench = get_best_lineup(squad_pdf)
                
                st.subheader(f"Starting XI - {sel_club_tac}")
                fig_pitch = create_pitch_plot(starters)
                st.plotly_chart(fig_pitch, use_container_width=True)
                
                st.divider()
                st.subheader(f"Bench ({len(bench)} players)")
                if not bench.empty:
                    if "value_eur" in bench.columns: bench["Value"] = bench["value_eur"].apply(format_currency_custom)
                    
                    b_cols = ["short_name", "player_positions", "overall", "age", "Value"]
                    b_cols = [c for c in b_cols if c in bench.columns]
                    
                    st.dataframe(bench[b_cols], use_container_width=True, hide_index=True, column_config={"overall": st.column_config.ProgressColumn("OVR", min_value=0, max_value=100, format="%d"), "short_name": "Player", "player_positions": "Pos"})
                else: st.info("No players on the bench.")
            else: st.error("No players found for this team in this year.")