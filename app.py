import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pyspark.sql.functions import col, lower, desc, avg, sum as _sum
from utils import load_all_data
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="FIFA Scout Pro", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
    }
    @media (prefers-color-scheme: dark) {
        .metric-box {
            background-color: #262730;
            border-color: #444;
        }
        .metric-value { color: #eee; }
        .metric-label { color: #aaa; }
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Advanced Scouting"

def set_page(page_name):
    st.session_state['current_page'] = page_name
    st.rerun()

# --- DEFINIZIONE MODULI (GLOBAL VAR) ---
# È importante che questa variabile sia definita QUI, all'inizio.
FORMATIONS = {
    "4-3-3": {
        "roles": ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    },
    "4-4-2": {
        "roles": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    },
    "4-2-3-1": {
        "roles": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "CAM", "LM", "RM", "ST"]
    },
    "3-5-2": {
        "roles": ["GK", "CB", "CB", "CB", "LM", "CDM", "CM", "CDM", "RM", "ST", "ST"]
    },
    "3-4-3": {
        "roles": ["GK", "CB", "CB", "CB", "LM", "CM", "CM", "RM", "LW", "ST", "RW"]
    }
}

# --- TACTICS MAP (COORDINATE GRAFICHE) ---
TACTICS_MAP = {
    "4-3-3": [
        {"role": "GK", "x": 5, "y": 50, "search": ["GK"]},
        {"role": "LB", "x": 25, "y": 90, "search": ["LB", "LWB"]},
        {"role": "LCB", "x": 25, "y": 62, "search": ["CB"]},
        {"role": "RCB", "x": 25, "y": 38, "search": ["CB"]},
        {"role": "RB", "x": 25, "y": 10, "search": ["RB", "RWB"]},
        {"role": "LCM", "x": 55, "y": 70, "search": ["CM", "CDM", "CAM"]},
        {"role": "CDM", "x": 50, "y": 50, "search": ["CDM", "CM"]},
        {"role": "RCM", "x": 55, "y": 30, "search": ["CM", "CDM", "CAM"]},
        {"role": "LW", "x": 85, "y": 85, "search": ["LW", "LM", "LF"]},
        {"role": "ST", "x": 90, "y": 50, "search": ["ST", "CF"]},
        {"role": "RW", "x": 85, "y": 15, "search": ["RW", "RM", "RF"]},
    ],
    "4-4-2": [
        {"role": "GK", "x": 5, "y": 50, "search": ["GK"]},
        {"role": "LB", "x": 25, "y": 90, "search": ["LB", "LWB"]},
        {"role": "LCB", "x": 25, "y": 62, "search": ["CB"]},
        {"role": "RCB", "x": 25, "y": 38, "search": ["CB"]},
        {"role": "RB", "x": 25, "y": 10, "search": ["RB", "RWB"]},
        {"role": "LM", "x": 60, "y": 90, "search": ["LM", "LW"]},
        {"role": "LCM", "x": 50, "y": 60, "search": ["CM", "CDM"]},
        {"role": "RCM", "x": 50, "y": 40, "search": ["CM", "CDM"]},
        {"role": "RM", "x": 60, "y": 10, "search": ["RM", "RW"]},
        {"role": "LST", "x": 85, "y": 60, "search": ["ST", "CF"]},
        {"role": "RST", "x": 85, "y": 40, "search": ["ST", "CF"]},
    ],
    "4-2-3-1": [
        {"role": "GK", "x": 5, "y": 50, "search": ["GK"]},
        {"role": "LB", "x": 25, "y": 90, "search": ["LB", "LWB"]},
        {"role": "LCB", "x": 25, "y": 62, "search": ["CB"]},
        {"role": "RCB", "x": 25, "y": 38, "search": ["CB"]},
        {"role": "RB", "x": 25, "y": 10, "search": ["RB", "RWB"]},
        {"role": "LDM", "x": 45, "y": 65, "search": ["CDM", "CM"]},
        {"role": "RDM", "x": 45, "y": 35, "search": ["CDM", "CM"]},
        {"role": "LAM", "x": 70, "y": 85, "search": ["LM", "LW", "CAM"]},
        {"role": "CAM", "x": 70, "y": 50, "search": ["CAM", "CF", "CM"]},
        {"role": "RAM", "x": 70, "y": 15, "search": ["RM", "RW", "CAM"]},
        {"role": "ST", "x": 90, "y": 50, "search": ["ST", "CF"]},
    ],
    "3-5-2": [
        {"role": "GK", "x": 5, "y": 50, "search": ["GK"]},
        {"role": "LCB", "x": 25, "y": 80, "search": ["CB"]},
        {"role": "CB", "x": 20, "y": 50, "search": ["CB"]},
        {"role": "RCB", "x": 25, "y": 20, "search": ["CB"]},
        {"role": "LM", "x": 55, "y": 90, "search": ["LM", "LW", "LWB"]},
        {"role": "LCM", "x": 50, "y": 65, "search": ["CM", "CDM"]},
        {"role": "CAM", "x": 65, "y": 50, "search": ["CAM", "CM"]},
        {"role": "RCM", "x": 50, "y": 35, "search": ["CM", "CDM"]},
        {"role": "RM", "x": 55, "y": 10, "search": ["RM", "RW", "RWB"]},
        {"role": "LST", "x": 85, "y": 60, "search": ["ST", "CF"]},
        {"role": "RST", "x": 85, "y": 40, "search": ["ST", "CF"]},
    ],
    "3-4-3": [
        {"role": "GK", "x": 5, "y": 50, "search": ["GK"]},
        {"role": "LCB", "x": 25, "y": 80, "search": ["CB"]},
        {"role": "CB", "x": 20, "y": 50, "search": ["CB"]},
        {"role": "RCB", "x": 25, "y": 20, "search": ["CB"]},
        {"role": "LM", "x": 55, "y": 90, "search": ["LM", "LW", "LWB"]},
        {"role": "LCM", "x": 50, "y": 65, "search": ["CM", "CDM"]},
        {"role": "RCM", "x": 50, "y": 35, "search": ["CM", "CDM"]},
        {"role": "RM", "x": 55, "y": 10, "search": ["RM", "RW", "RWB"]},
        {"role": "LW", "x": 85, "y": 85, "search": ["LW", "LM", "LF"]},
        {"role": "ST", "x": 90, "y": 50, "search": ["ST", "CF"]},
        {"role": "RW", "x": 85, "y": 15, "search": ["RW", "RM", "RF"]},
    ]
}

# --- UTILS ---
def format_currency_custom(val):
    if pd.isna(val) or val == 0: return "0"
    if val >= 1_000_000:
        return f"EUR {val/1_000_000:.1f}M"
    return f"EUR {val:,.0f}"

# --- 1. DATA LOADING ---
@st.cache_resource(show_spinner="Loading Engine...")
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
    c_pdf = _df_coaches.select("coach_id", col("short_name").alias("coach_name")).distinct().toPandas()
    coach_map = dict(zip(c_pdf.coach_id, c_pdf.coach_name))
    
    t_col = "club_name" if "club_name" in _df_teams.columns else "team_name"
    t_pdf = _df_teams.select(col(t_col).alias("club_name"), col("league_name").alias("league_ref")).distinct().toPandas()
    team_to_league = dict(zip(t_pdf.club_name, t_pdf.league_ref))
    
    t_c_pdf = _df_teams.select(col(t_col).alias("club_name"), "coach_id").distinct().toPandas()
    team_to_coach_id = dict(zip(t_c_pdf.club_name, t_c_pdf.coach_id))
    
    return coach_map, team_to_league, team_to_coach_id

with st.spinner("Preparing Data..."):
    coach_map, team_to_league, team_to_coach_id = get_lookups(df_teams, df_coaches)

all_leagues = sorted(list(set([l for l in team_to_league.values() if l])))
if "fifa_version" in df_players.columns:
    all_versions = [int(r[0]) for r in df_players.select("fifa_version").distinct().sort(desc("fifa_version")).collect()]
else: all_versions = []

all_works = [r[0] for r in df_players.select("work_rate").distinct().dropna().collect()] if "work_rate" in df_players.columns else []

# --- LOGICA FORMAZIONE (GEOMETRICA) ---
def get_best_lineup(df, module_name="4-3-3"):
    df = df.copy()
    df = df.sort_values('overall', ascending=False)
    
    # Usa TACTICS_MAP per definire i ruoli esatti
    scheme = TACTICS_MAP.get(module_name, TACTICS_MAP["4-3-3"])
    
    starters = []
    used_indices = []
    
    for slot in scheme:
        target_role = slot['search']
        
        found_idx = None
        found_player = None
        
        # 1. Match Esatto
        for idx, row in df.iterrows():
            if idx in used_indices: continue
            p_pos_list = [p.strip() for p in row['player_positions'].split(',')] if row['player_positions'] else []
            
            if any(role in p_pos_list for role in target_role):
                found_idx = idx
                found_player = row.to_dict()
                break
        
        # 2. Fallback (Adatta)
        if found_player is None:
            is_gk_slot = "GK" in target_role
            is_def_slot = any(x in target_role for x in ["LB","RB","CB","LWB","RWB"])
            
            for idx, row in df.iterrows():
                if idx in used_indices: continue
                p_pos_str = row['player_positions'] if row['player_positions'] else ""
                
                if is_gk_slot:
                    if "GK" in p_pos_str: 
                        found_idx = idx; found_player = row.to_dict(); break
                else:
                    if "GK" in p_pos_str: continue 
                    if is_def_slot:
                        if any(x in p_pos_str for x in ["B", "CB"]):
                            found_idx = idx; found_player = row.to_dict(); break
                    else:
                        found_idx = idx; found_player = row.to_dict(); break

        if found_player:
            found_player['tactical_role'] = slot['role']
            found_player['x'] = slot['x']
            found_player['y'] = slot['y']
            
            starters.append(found_player)
            used_indices.append(found_idx)
            
    starters_df = pd.DataFrame(starters)
    bench_df = df.drop(index=used_indices)
    
    return starters_df, bench_df

# --- PLOT PITCH (FISSO) ---
def create_pitch_plot(starters_df):
    fig = go.Figure()
    
    field_shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, layer="below", line=dict(width=0), fillcolor="#2e7d32"),
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="circle", x0=40, y0=40, x1=60, y1=60, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="rect", x0=0, y0=20, x1=17, y1=80, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="rect", x0=83, y0=20, x1=100, y1=80, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="rect", x0=-2, y0=45, x1=0, y1=55, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
        dict(type="rect", x0=100, y0=45, x1=102, y1=55, layer="below", line=dict(color="rgba(255,255,255,0.7)", width=2)),
    ]
    fig.update_layout(shapes=field_shapes)
    
    for _, player in starters_df.iterrows():
        hover_text = f"<b>{player['short_name']}</b><br>Role: {player['tactical_role']}<br>OVR: {player['overall']}"
        
        fig.add_trace(go.Scatter(
            x=[player['x']], y=[player['y']],
            mode='markers+text',
            marker=dict(size=28, color='white', line=dict(color='black', width=2)),
            text=str(player['overall']),
            textposition="middle center",
            textfont=dict(color='black', size=11, family="Arial Black"),
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[player['x']], y=[player['y']-8],
            mode='text',
            text=f"<b>{player['short_name']}</b>",
            textposition="bottom center",
            textfont=dict(color='white', size=12, family="Arial", shadow="1px 1px 2px black"),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-5, 105]), 
        yaxis=dict(visible=False, range=[0, 100]), 
        plot_bgcolor="#2e7d32", 
        margin=dict(l=10, r=10, t=10, b=10), 
        height=650, 
        dragmode=False
    )
    return fig

def calculate_team_stats(df_team):
    pdf = df_team.toPandas()
    if pdf.empty: return {}
    def get_simple_role(pos):
        if not pos: return "MID"
        main = pos.split(',')[0]
        if 'GK' in main: return "GK"
        if any(x in main for x in ['B', 'CB']): return "DEF"
        if any(x in main for x in ['M', 'CDM', 'CAM']): return "MID"
        return "FWD"
    pdf['macro'] = pdf['player_positions'].apply(get_simple_role)
    stats = {
        "Overall Avg": pdf['overall'].mean(),
        "Attack Avg": pdf[pdf['macro'] == 'FWD']['overall'].mean() if not pdf[pdf['macro'] == 'FWD'].empty else 0,
        "Midfield Avg": pdf[pdf['macro'] == 'MID']['overall'].mean() if not pdf[pdf['macro'] == 'MID'].empty else 0,
        "Defense Avg": pdf[pdf['macro'] == 'DEF']['overall'].mean() if not pdf[pdf['macro'] == 'DEF'].empty else 0,
        "Age Avg": pdf['age'].mean(),
        "Total Value": pdf['value_eur'].sum(),
        "Count": len(pdf)
    }
    return stats

# --- HISTORY FUNCTIONS ---
def get_player_history(name):
    hist_df = df_players.filter(lower(col("short_name")) == name.lower()).select("fifa_version", "overall", "value_eur").orderBy("fifa_version").toPandas()
    return hist_df

def get_team_history(team_name):
    hist_team = df_players.filter(col("club_name") == team_name).groupBy("fifa_version") \
        .agg(avg("overall").alias("avg_ovr"), _sum("value_eur").alias("tot_val")) \
        .orderBy("fifa_version").toPandas()
    return hist_team


# =================================================================================
# SIDEBAR NAVIGATION
# =================================================================================
with st.sidebar:
    st.title("FIFA Scout")
    st.markdown("Select a tool:")
    
    # 5 BOTTONI
    type_scout = "primary" if st.session_state['current_page'] == "Advanced Scouting" else "secondary"
    if st.button("Advanced Scouting", type=type_scout): set_page("Advanced Scouting")
        
    type_comp = "primary" if st.session_state['current_page'] == "Player Comparison" else "secondary"
    if st.button("Player Comparison", type=type_comp): set_page("Player Comparison")
        
    type_tcomp = "primary" if st.session_state['current_page'] == "Team Comparison" else "secondary"
    if st.button("Team Comparison", type=type_tcomp): set_page("Team Comparison")

    type_tac = "primary" if st.session_state['current_page'] == "Team Tactics" else "secondary"
    if st.button("Team Tactics", type=type_tac): set_page("Team Tactics")
    
    type_best = "primary" if st.session_state['current_page'] == "Best XI History" else "secondary"
    if st.button("Best XI History", type=type_best): set_page("Best XI History")
        
    st.divider()
    st.info("Data based on EA FC Database")


# =================================================================================
# PAGE ROUTER
# =================================================================================

# --------------------------
# PAGE 1: ADVANCED SCOUTING
# --------------------------
if st.session_state['current_page'] == "Advanced Scouting":
    st.title("Advanced Scouting Engine")
    
    st.sidebar.markdown("### Filters")
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
            if "club_contract_valid_until_year" in df_players.columns: max_contract = st.slider("Contract Expiring By", 2023, 2050, 2032)
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
            if min_team_rating > 50: pass
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
        pdf_view["Coach ID"] = pdf_view["club_name"].map(team_to_coach_id)
        pdf_view["Coach"] = pdf_view["Coach ID"].map(coach_map)
        if "fifa_version" in pdf_view.columns: pdf_view["Edition"] = pdf_view["fifa_version"].apply(lambda x: f"FIFA {int(x)}" if pd.notnull(x) else "")
        if "value_eur" in pdf_view.columns: pdf_view["ValueFormatted"] = pdf_view["value_eur"].apply(format_currency_custom)
        
        m1, m2, m3, m4 = st.columns(4)
        avg_ovr = pdf_view['overall'].mean()
        avg_pot = pdf_view['potential'].mean()
        avg_val = pdf_view['value_eur'].mean() if 'value_eur' in pdf_view.columns else 0
        count_p = len(pdf_view)
        
        m1.markdown(f"<div class='metric-box'><div class='metric-value'>{count_p}</div><div class='metric-label'>Players Found</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_ovr:.1f}</div><div class='metric-label'>Avg Overall</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_pot:.1f}</div><div class='metric-label'>Avg Potential</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-box'><div class='metric-value'>{format_currency_custom(avg_val)}</div><div class='metric-label'>Avg Value</div></div>", unsafe_allow_html=True)
        
        st.divider()

        tab_list, tab_charts = st.tabs(["List View", "Analytics Charts"])
        
        with tab_list:
            csv = pdf_view.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="scouting_results.csv", mime="text/csv", type="primary")
            
            display_cols = ["short_name", "Edition", "age", "overall", "potential", "club_name", "League", "Coach", "ValueFormatted"]
            display_cols = [c for c in display_cols if c in pdf_view.columns]
            st.dataframe(pdf_view[display_cols], use_container_width=True, height=600, hide_index=True, column_config={"short_name": "Name", "club_name": "Club", "age": "Age", "overall": st.column_config.ProgressColumn("Overall", format="%d", min_value=0, max_value=100)})

            st.divider()
            st.markdown("### Player History Analysis")
            trend_player = st.selectbox("Select a player from results to see history trend:", pdf_view['short_name'].unique())
            if trend_player:
                hist_df = get_player_history(trend_player)
                if not hist_df.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_val = px.line(hist_df, x="fifa_version", y="value_eur", title=f"{trend_player} - Market Value Trajectory", markers=True)
                        st.plotly_chart(fig_val, use_container_width=True)
                    with c2:
                        fig_ovr = px.line(hist_df, x="fifa_version", y="overall", title=f"{trend_player} - Overall History", markers=True)
                        st.plotly_chart(fig_ovr, use_container_width=True)

        with tab_charts:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Overall Distribution")
                fig_hist = px.histogram(pdf_view, x="overall", nbins=20, color_discrete_sequence=['#4CAF50'])
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                st.markdown("#### Age Distribution")
                fig_age = px.histogram(pdf_view, x="age", nbins=15, color_discrete_sequence=['#2196F3'])
                fig_age.update_layout(bargap=0.1)
                st.plotly_chart(fig_age, use_container_width=True)
                
    elif submitted: st.warning("No players found matching your criteria.")
    else: st.info("Set filters on the left and click SEARCH NOW.")

# --------------------------
# PAGE 2: PLAYER COMPARISON
# --------------------------
elif st.session_state['current_page'] == "Player Comparison":
    st.title("Head-to-Head Comparison")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Player A")
        search_p1 = st.text_input("Search Name (Player A)", key="s1")
        ver_p1 = st.selectbox("Version (A)", all_versions, index=0, key="v1") if all_versions else None
        cand1 = pd.DataFrame()
        if search_p1:
            f1 = df_players.filter(lower(col("short_name")).contains(search_p1.lower()))
            if ver_p1: f1 = f1.filter(col("fifa_version") == ver_p1)
            cand1 = f1.select("short_name", "club_name", "overall").limit(5).toPandas()
        opts1 = [f"{r['short_name']} ({r['club_name']}) - {r['overall']}" for _, r in cand1.iterrows()] if not cand1.empty else []
        sel_p1 = st.selectbox("Select Player A", opts1, key="sel1")

    with c2:
        st.markdown("### Player B")
        search_p2 = st.text_input("Search Name (Player B)", key="s2")
        ver_p2 = st.selectbox("Version (B)", all_versions, index=0, key="v2") if all_versions else None
        cand2 = pd.DataFrame()
        if search_p2:
            f2 = df_players.filter(lower(col("short_name")).contains(search_p2.lower()))
            if ver_p2: f2 = f2.filter(col("fifa_version") == ver_p2)
            cand2 = f2.select("short_name", "club_name", "overall").limit(5).toPandas()
        opts2 = [f"{r['short_name']} ({r['club_name']}) - {r['overall']}" for _, r in cand2.iterrows()] if not cand2.empty else []
        sel_p2 = st.selectbox("Select Player B", opts2, key="sel2")

    if st.button("COMPARE PLAYERS", type="primary", use_container_width=True):
        if sel_p1 and sel_p2:
            name1 = sel_p1.split(" (")[0]
            name2 = sel_p2.split(" (")[0]
            stats_cols = ["short_name", "overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age", "height_cm", "weight_kg", "weak_foot", "skill_moves"]
            stats_cols = [c for c in stats_cols if c in df_players.columns]
            p1_data = df_players.filter((col("short_name") == name1) & (col("fifa_version") == ver_p1)).select(stats_cols).toPandas().iloc[0]
            p2_data = df_players.filter((col("short_name") == name2) & (col("fifa_version") == ver_p2)).select(stats_cols).toPandas().iloc[0]
            
            st.divider()
            metrics = [("Overall Rating", "overall"), ("Potential", "potential"), ("Pace", "pace"), ("Shooting", "shooting"), ("Passing", "passing"), ("Dribbling", "dribbling"), ("Defending", "defending"), ("Physicality", "physic"), ("Age", "age"), ("Height (cm)", "height_cm"), ("Weight (kg)", "weight_kg"), ("Weak Foot", "weak_foot"), ("Skill Moves", "skill_moves")]
            h1, h2, h3 = st.columns([1, 0.5, 1])
            h1.markdown(f"<h3 style='text-align: center;'>{p1_data['short_name']}</h3>", unsafe_allow_html=True)
            h2.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
            h3.markdown(f"<h3 style='text-align: center;'>{p2_data['short_name']}</h3>", unsafe_allow_html=True)
            st.markdown("---")
            for label, key in metrics:
                if key not in p1_data or key not in p2_data: continue
                val1 = p1_data[key]; val2 = p2_data[key]
                if pd.isna(val1): val1 = 0
                if pd.isna(val2): val2 = 0
                diff = val1 - val2
                color1 = "green" if val1 > val2 else "white"; weight1 = "bold" if val1 > val2 else "normal"
                color2 = "green" if val2 > val1 else "white"; weight2 = "bold" if val2 > val1 else "normal"
                if diff > 0: diff_str = f"+{int(diff)}"
                elif diff < 0: diff_str = f"{int(diff)}"
                else: diff_str = "="
                rc1, rc2, rc3 = st.columns([1, 0.5, 1])
                with rc1: st.markdown(f"<div style='text-align: center; color: {color1}; font-weight: {weight1}; font-size: 18px;'>{int(val1)}</div><div style='text-align: center; font-size: 12px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                with rc2: st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 10px;'>{diff_str}</div>", unsafe_allow_html=True)
                with rc3: st.markdown(f"<div style='text-align: center; color: {color2}; font-weight: {weight2}; font-size: 18px;'>{int(val2)}</div><div style='text-align: center; font-size: 12px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                st.divider()
            
            st.subheader("Radar Comparison")
            radar_cats = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            fig = go.Figure()
            vals1 = [p1_data[c] for c in radar_cats]; vals1 += [vals1[0]]
            fig.add_trace(go.Scatterpolar(r=vals1, theta=radar_cats + [radar_cats[0]], fill='toself', name=p1_data['short_name']))
            vals2 = [p2_data[c] for c in radar_cats]; vals2 += [vals2[0]]
            fig.add_trace(go.Scatterpolar(r=vals2, theta=radar_cats + [radar_cats[0]], fill='toself', name=p2_data['short_name']))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Historical Comparison")
            hist1 = get_player_history(name1)
            hist2 = get_player_history(name2)
            hist1['Player'] = name1
            hist2['Player'] = name2
            combined_hist = pd.concat([hist1, hist2])
            
            c1, c2 = st.columns(2)
            with c1:
                fig_trend_val = px.line(combined_hist, x="fifa_version", y="value_eur", color="Player", title="Market Value Trajectory", markers=True)
                st.plotly_chart(fig_trend_val, use_container_width=True)
            with c2:
                fig_trend_ovr = px.line(combined_hist, x="fifa_version", y="overall", color="Player", title="Overall Rating Trajectory", markers=True)
                st.plotly_chart(fig_trend_ovr, use_container_width=True)
            
        else:
            st.warning("Please select both players to compare.")

# --------------------------
# PAGE 3: TEAM COMPARISON
# --------------------------
elif st.session_state['current_page'] == "Team Comparison":
    st.title("Team Analysis & Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Team A")
        l1 = st.selectbox("League A", [""] + all_leagues, key="l1")
        teams1 = sorted([t for t, l in team_to_league.items() if l == l1]) if l1 else []
        t1 = st.selectbox("Club A", teams1, key="t1")
        v1 = st.selectbox("Version A", all_versions, index=0, key="v_t1") if all_versions else None
    with col2:
        st.markdown("### Team B")
        l2 = st.selectbox("League B", [""] + all_leagues, key="l2")
        teams2 = sorted([t for t, l in team_to_league.items() if l == l2]) if l2 else []
        t2 = st.selectbox("Club B", teams2, key="t2")
        v2 = st.selectbox("Version B", all_versions, index=0, key="v_t2") if all_versions else None
        
    if st.button("COMPARE TEAMS", type="primary"):
        if t1 and t2:
            df_a = df_players.filter((col("club_name") == t1) & (col("fifa_version") == v1))
            stats_a = calculate_team_stats(df_a)
            df_b = df_players.filter((col("club_name") == t2) & (col("fifa_version") == v2))
            stats_b = calculate_team_stats(df_b)
            if stats_a and stats_b:
                st.divider()
                metrics = [("Overall Rating", "Overall Avg", ".1f"), ("Attack Rating", "Attack Avg", ".1f"), ("Midfield Rating", "Midfield Avg", ".1f"), ("Defense Rating", "Defense Avg", ".1f"), ("Average Age", "Age Avg", ".1f"), ("Total Value", "Total Value", "currency")]
                h1, h2, h3 = st.columns([1, 0.5, 1])
                h1.markdown(f"<h3 style='text-align: center;'>{t1}</h3>", unsafe_allow_html=True)
                h2.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
                h3.markdown(f"<h3 style='text-align: center;'>{t2}</h3>", unsafe_allow_html=True)
                st.markdown("---")
                for label, key, fmt in metrics:
                    val_a = stats_a.get(key, 0); val_b = stats_b.get(key, 0)
                    diff = val_a - val_b
                    if val_a > val_b: c_a, w_a = "green", "bold"; c_b, w_b = "white", "normal"
                    elif val_b > val_a: c_a, w_a = "white", "normal"; c_b, w_b = "green", "bold"
                    else: c_a = c_b = "white"; w_a = w_b = "normal"
                    if fmt == "currency":
                        str_a = format_currency_custom(val_a); str_b = format_currency_custom(val_b); diff_str = ""
                    else:
                        str_a = f"{val_a:.1f}"; str_b = f"{val_b:.1f}"
                        if diff > 0: diff_str = f"+{diff:.1f}"
                        elif diff < 0: diff_str = f"{diff:.1f}"
                        else: diff_str = "="
                    rc1, rc2, rc3 = st.columns([1, 0.5, 1])
                    with rc1: st.markdown(f"<div style='text-align: center; color: {c_a}; font-weight: {w_a}; font-size: 20px;'>{str_a}</div><div style='text-align: center; font-size: 12px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                    with rc2: st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 10px;'>{diff_str}</div>", unsafe_allow_html=True)
                    with rc3: st.markdown(f"<div style='text-align: center; color: {c_b}; font-weight: {w_b}; font-size: 20px;'>{str_b}</div><div style='text-align: center; font-size: 12px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                    st.divider()
                st.subheader("Team Radar Comparison")
                radar_cats = ["Attack Avg", "Midfield Avg", "Defense Avg", "Overall Avg"]
                fig = go.Figure()
                vals_a = [stats_a[k] for k in radar_cats]; vals_a += [vals_a[0]]
                fig.add_trace(go.Scatterpolar(r=vals_a, theta=[k.replace(" Avg", "") for k in radar_cats] + [radar_cats[0].replace(" Avg", "")], fill='toself', name=t1))
                vals_b = [stats_b[k] for k in radar_cats]; vals_b += [vals_b[0]]
                fig.add_trace(go.Scatterpolar(r=vals_b, theta=[k.replace(" Avg", "") for k in radar_cats] + [radar_cats[0].replace(" Avg", "")], fill='toself', name=t2))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[50, 95])), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- CLUB HISTORY ---
                st.divider()
                st.subheader("Club History Analysis")
                hist_t1 = get_team_history(t1); hist_t2 = get_team_history(t2)
                hist_t1['Club'] = t1; hist_t2['Club'] = t2
                combined_team_hist = pd.concat([hist_t1, hist_t2])
                c1, c2 = st.columns(2)
                with c1:
                    fig_team_val = px.line(combined_team_hist, x="fifa_version", y="tot_val", color="Club", title="Total Squad Value History", markers=True)
                    st.plotly_chart(fig_team_val, use_container_width=True)
                with c2:
                    fig_team_ovr = px.line(combined_team_hist, x="fifa_version", y="avg_ovr", color="Club", title="Average Squad Rating History", markers=True)
                    st.plotly_chart(fig_team_ovr, use_container_width=True)
            else: st.error("Could not calculate stats for one of the teams.")
        else: st.warning("Please select both teams.")

# --------------------------
# PAGE 4: TEAM TACTICS
# --------------------------
elif st.session_state['current_page'] == "Team Tactics":
    st.title("Team Squad Builder")
    col_l, col_t, col_y = st.columns(3)
    sel_league_tac = col_l.selectbox("Select League", all_leagues)
    clubs_in_league = sorted([t for t, l in team_to_league.items() if l == sel_league_tac])
    sel_club_tac = col_t.selectbox("Select Club", clubs_in_league)
    sel_year_tac = col_y.selectbox("Select Year", all_versions, index=0) if all_versions else None
    st.divider()
    col_mod, _ = st.columns([1, 2])
    sel_formation = col_mod.selectbox("Select Tactical Module", list(TACTICS_MAP.keys()), index=0)
    
    if st.button("Show Squad", type="primary"):
        with st.spinner(f"Scouting {sel_club_tac} ({sel_year_tac}) - Module {sel_formation}..."):
            df_squad = df_players.filter((col("club_name") == sel_club_tac) & (col("fifa_version") == sel_year_tac))
            cols_needed = ["short_name", "player_positions", "overall", "potential", "age", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
            cols_needed = [c for c in cols_needed if c in df_players.columns]
            squad_pdf = df_squad.select(cols_needed).orderBy(desc("overall")).toPandas()
            if not squad_pdf.empty:
                starters, bench = get_best_lineup(squad_pdf, module_name=sel_formation)
                
                st.markdown("### Tactical Analysis")
                avg_ovr = starters['overall'].mean()
                avg_age = starters['age'].mean()
                tot_val = starters['value_eur'].sum() if 'value_eur' in starters.columns else 0
                k1, k2, k3 = st.columns(3)
                k1.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_ovr:.1f}</div><div class='metric-label'>Squad Rating</div></div>", unsafe_allow_html=True)
                k2.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_age:.1f}</div><div class='metric-label'>Average Age</div></div>", unsafe_allow_html=True)
                k3.markdown(f"<div class='metric-box'><div class='metric-value'>{format_currency_custom(tot_val)}</div><div class='metric-label'>Starting XI Value</div></div>", unsafe_allow_html=True)
                st.divider()
                
                st.subheader(f"Starting XI - {sel_club_tac} ({sel_formation})")
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

# --------------------------
# PAGE 5: BEST XI HISTORY
# --------------------------
elif st.session_state['current_page'] == "Best XI History":
    st.title("Best XI of the Year")
    st.markdown("Select a FIFA Edition to see the strongest possible lineup.")
    
    col_ver, col_mod = st.columns(2)
    sel_best_ver = col_ver.selectbox("Select FIFA Version", all_versions)
    sel_best_mod = col_mod.selectbox("Select Formation", list(TACTICS_MAP.keys()), index=0)
    
    if st.button("GENERATE BEST XI", type="primary"):
        with st.spinner(f"Finding best players in {sel_best_ver}..."):
            df_year = df_players.filter(col("fifa_version") == sel_best_ver)
            cols_needed = ["short_name", "player_positions", "overall", "potential", "age", "value_eur", "club_name"]
            cols_needed = [c for c in cols_needed if c in df_players.columns]
            top_pdf = df_year.select(cols_needed).orderBy(desc("overall")).limit(2000).toPandas()
            
            if not top_pdf.empty:
                starters, bench = get_best_lineup(top_pdf, module_name=sel_best_mod)
                
                # --- SUMMARY STATS (NEW) ---
                avg_ovr = starters['overall'].mean()
                avg_age = starters['age'].mean()
                tot_val = starters['value_eur'].sum() if 'value_eur' in starters.columns else 0
                
                s1, s2, s3 = st.columns(3)
                s1.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_ovr:.1f}</div><div class='metric-label'>Squad Rating</div></div>", unsafe_allow_html=True)
                s2.markdown(f"<div class='metric-box'><div class='metric-value'>{avg_age:.1f}</div><div class='metric-label'>Average Age</div></div>", unsafe_allow_html=True)
                s3.markdown(f"<div class='metric-box'><div class='metric-value'>{format_currency_custom(tot_val)}</div><div class='metric-label'>Total Market Value</div></div>", unsafe_allow_html=True)
                st.divider()

                st.subheader(f"World Best XI - FIFA {sel_best_ver}")
                fig_pitch = create_pitch_plot(starters)
                st.plotly_chart(fig_pitch, use_container_width=True)
                
                st.divider()
                st.subheader("Starters Details")
                disp_cols = ["short_name", "club_name", "overall", "player_positions"]
                st.dataframe(starters[disp_cols], use_container_width=True, hide_index=True, column_config={"overall": st.column_config.ProgressColumn("OVR", format="%d", min_value=80, max_value=100)})
                
            else:
                st.error("No data found for this version.")
