import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pyspark.sql.functions import col, lower, desc
from utils import load_all_data
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="FIFA Scout", layout="wide")
st.title("FIFA Scout")

# --- DEFINIZIONE MODULI ---
FORMATIONS = {
    "4-3-3": {"DEF": ["LB", "CB", "CB", "RB"], "MID": ["CM", "CDM", "CM"], "FWD": ["LW", "ST", "RW"]},
    "4-4-2": {"DEF": ["LB", "CB", "CB", "RB"], "MID": ["LM", "CM", "CM", "RM"], "FWD": ["ST", "ST"]},
    "4-2-3-1": {"DEF": ["LB", "CB", "CB", "RB"], "MID": ["CDM", "CDM", "CAM", "LM", "RM"], "FWD": ["ST"]},
    "3-5-2": {"DEF": ["CB", "CB", "CB"], "MID": ["LM", "CDM", "CM", "CDM", "RM"], "FWD": ["ST", "ST"]},
    "3-4-3": {"DEF": ["CB", "CB", "CB"], "MID": ["LM", "CM", "CM", "RM"], "FWD": ["LW", "ST", "RW"]}
}

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

with st.spinner("Indexing..."):
    coach_map, team_to_league, team_to_coach_id = get_lookups(df_teams, df_coaches)

all_leagues = sorted(list(set([l for l in team_to_league.values() if l])))
all_works = [r[0] for r in df_players.select("work_rate").distinct().dropna().collect()] if "work_rate" in df_players.columns else []

if "fifa_version" in df_players.columns:
    all_versions = [int(r[0]) for r in df_players.select("fifa_version").distinct().sort(desc("fifa_version")).collect()]
else: all_versions = []

# --- LOGICA FORMAZIONE ---
def get_best_lineup(df, module_name="4-3-3"):
    df = df.copy()
    def get_role_group(pos):
        if not pos: return "MID"
        main = pos.split(',')[0]
        if 'GK' in main: return 'GK'
        if any(x in main for x in ['B', 'CB', 'WB', 'LB', 'RB']): return 'DEF'
        if any(x in main for x in ['M', 'CDM', 'CAM', 'CM', 'LM', 'RM']): return 'MID'
        return 'FWD'

    df['role_group'] = df['player_positions'].apply(get_role_group)
    df = df.sort_values('overall', ascending=False)
    
    # 1. Separazione GK
    df['is_gk'] = df['player_positions'].apply(lambda x: 'GK' in x.split(',')[0] if x else False)
    gk_pool = df[df['is_gk'] == True]
    outfield_pool = df[df['is_gk'] == False]
    
    starters = []
    if not gk_pool.empty: starters.append(gk_pool.iloc[0])
    
    config = FORMATIONS.get(module_name, FORMATIONS["4-3-3"])
    target_slots = config["DEF"] + config["MID"] + config["FWD"]
    selected_ids = [starters[0]['short_name']] if starters else []
    
    def player_can_play(pos_str, target_role):
        if target_role == "CB" and ("CB" in pos_str): return True
        if target_role in ["LB", "LWB"] and any(x in pos_str for x in ["LB", "LWB"]): return True
        if target_role in ["RB", "RWB"] and any(x in pos_str for x in ["RB", "RWB"]): return True
        if target_role in ["CDM", "CM", "CAM"] and any(x in pos_str for x in ["M", "CDM", "CAM"]): return True
        if target_role in ["LM", "LW"] and any(x in pos_str for x in ["LM", "LW"]): return True
        if target_role in ["RM", "RW"] and any(x in pos_str for x in ["RM", "RW"]): return True
        if target_role == "ST" and any(x in pos_str for x in ["ST", "CF"]): return True
        return False

    for target in target_slots:
        candidates = outfield_pool[~outfield_pool['short_name'].isin(selected_ids)]
        match = None
        for _, p in candidates.iterrows():
            if player_can_play(p['player_positions'], target):
                match = p
                break
        
        if match is not None:
            starters.append(match)
            selected_ids.append(match['short_name'])
        else:
            if not candidates.empty:
                best_remaining = candidates.iloc[0]
                starters.append(best_remaining)
                selected_ids.append(best_remaining['short_name'])

    starters_df = pd.DataFrame(starters)
    bench_df = df[~df['short_name'].isin(selected_ids)]
    return starters_df, bench_df

# --- PLOT PITCH ---
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

    def get_macro_role(pos):
        if not pos: return "MID"
        main = pos.split(',')[0]
        if 'GK' in main: return 'GK'
        if any(x in main for x in ['B', 'CB', 'WB', 'LB', 'RB']): return 'DEF'
        if any(x in main for x in ['M', 'CDM', 'CAM', 'CM', 'LM', 'RM']): return 'MID'
        return 'FWD'

    def get_position_score(positions):
        if not positions: return 50
        main_pos = positions.split(',')[0]
        scores = {'LW': 90, 'LF': 90, 'LB': 90, 'ST': 50, 'CF': 50, 'CAM': 50, 'CM': 50, 'CDM': 50, 'CB': 50, 'GK': 50, 'RW': 10, 'RF': 10, 'RB': 10}
        return scores.get(main_pos, 50)

    roles = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
    for _, player in players_df.iterrows():
        r = get_macro_role(player['player_positions'])
        player['pos_score'] = get_position_score(player['player_positions'])
        roles[r].append(player)

    def get_y_coords(count):
        if count == 1: return [50]
        step = 100 / (count + 1)
        return [step * (i+1) for i in range(count)][::-1]

    for role, p_list in roles.items():
        if not p_list: continue
        p_list.sort(key=lambda x: x['pos_score'], reverse=True)
        ys = get_y_coords(len(p_list))
        x = x_positions[role]
        for i, player in enumerate(p_list):
            hover_text = f"<b>{player['short_name']}</b><br>OVR: {player['overall']}<br>Pos: {player['player_positions']}"
            fig.add_trace(go.Scatter(x=[x], y=[ys[i]], mode='markers+text', marker=dict(size=22, color='white', line=dict(color='black', width=2)), text=str(player['overall']), textposition="middle center", textfont=dict(color='black', size=11, family="Arial Black"), hoverinfo="text", hovertext=hover_text, showlegend=False))
            fig.add_trace(go.Scatter(x=[x], y=[ys[i]-8], mode='text', text=f"<b>{player['short_name']}</b>", textposition="bottom center", textfont=dict(color='white', size=11, family="Arial", shadow="2px 2px 2px black"), hoverinfo="skip", showlegend=False))

    fig.update_layout(xaxis=dict(visible=False, range=[-5, 105]), yaxis=dict(visible=False, range=[0, 100]), plot_bgcolor="#43a047", margin=dict(l=10, r=10, t=10, b=10), height=600, dragmode=False)
    return fig


# ==========================================
# MODE SELECTOR
# ==========================================
st.sidebar.title("Navigation")
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
                max_contract = st.slider("Contract Expiring By", 2023, 2032, 2032)
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
            val_range = st.slider("Max Value", 0, 150000000, 150000000, step=500000)
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
        if "value_eur" in pdf_view.columns: pdf_view["Value"] = pdf_view["value_eur"].apply(format_currency_custom)
        display_cols = ["short_name", "Edition", "age", "overall", "potential", "club_name", "League", "Coach", "Value"]
        display_cols = [c for c in display_cols if c in pdf_view.columns]
        st.subheader(f"Search Results: {len(pdf_view)}")
        st.dataframe(pdf_view[display_cols], use_container_width=True, height=700, hide_index=True, column_config={"short_name": "Name", "club_name": "Club", "age": "Age", "overall": st.column_config.ProgressColumn("Overall", format="%d", min_value=0, max_value=100)})
    elif submitted: st.warning("No players found matching your criteria.")
    else: st.info("Set filters on the left and click SEARCH NOW.")

# ==========================================
# MODE 2: PLAYER COMPARISON (HEAD TO HEAD)
# ==========================================
elif mode == "Player Comparison":
    st.markdown("## Head-to-Head Comparison")
    st.markdown("Select two players to see who wins in each category.")

    # Due colonne per i selettori
    c1, c2 = st.columns(2)
    
    # Ricerca Giocatore 1
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

    # Ricerca Giocatore 2
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

    # Bottone Confronto
    if st.button("COMPARE PLAYERS", type="primary"):
        if sel_p1 and sel_p2:
            name1 = sel_p1.split(" (")[0]
            name2 = sel_p2.split(" (")[0]
            
            # Fetch dati
            stats_cols = ["short_name", "overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age", "height_cm", "weight_kg", "weak_foot", "skill_moves"]
            stats_cols = [c for c in stats_cols if c in df_players.columns]
            
            p1_data = df_players.filter((col("short_name") == name1) & (col("fifa_version") == ver_p1)).select(stats_cols).toPandas().iloc[0]
            p2_data = df_players.filter((col("short_name") == name2) & (col("fifa_version") == ver_p2)).select(stats_cols).toPandas().iloc[0]
            
            st.divider()
            
            # --- CONFRONTO VISIVO ---
            # Definiamo le metriche da confrontare
            metrics = [
                ("Overall Rating", "overall"),
                ("Potential", "potential"),
                ("Pace", "pace"),
                ("Shooting", "shooting"),
                ("Passing", "passing"),
                ("Dribbling", "dribbling"),
                ("Defending", "defending"),
                ("Physicality", "physic"),
                ("Age", "age"), # Attenzione: per l'età, minore è meglio? Dipende. Qui trattiamo maggiore come numero più alto.
                ("Height (cm)", "height_cm"),
                ("Weight (kg)", "weight_kg"),
                ("Weak Foot", "weak_foot"),
                ("Skill Moves", "skill_moves")
            ]
            
            # Intestazione Colonne
            h1, h2, h3 = st.columns([1, 0.5, 1])
            h1.markdown(f"<h3 style='text-align: center;'>{p1_data['short_name']}</h3>", unsafe_allow_html=True)
            h2.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
            h3.markdown(f"<h3 style='text-align: center;'>{p2_data['short_name']}</h3>", unsafe_allow_html=True)
            
            st.markdown("---")

            # Loop per ogni statistica
            for label, key in metrics:
                if key not in p1_data or key not in p2_data: continue
                
                val1 = p1_data[key]
                val2 = p2_data[key]
                
                # Gestione valori nulli
                if pd.isna(val1): val1 = 0
                if pd.isna(val2): val2 = 0
                
                # Calcolo differenza
                diff = val1 - val2
                
                # Formattazione Colore
                # Se vince P1 -> P1 Verde, P2 Grigio
                # Se vince P2 -> P1 Grigio, P2 Verde
                # Se Pareggio -> Entrambi Grigio
                
                # Eccezione Età: In genere più giovane è meglio per potenziale, ma più vecchio per esperienza.
                # Qui trattiamo valore numerico puro: più alto vince il verde.
                
                color1 = "green" if val1 > val2 else "red"
                weight1 = "bold" if val1 > val2 else "normal"
                
                color2 = "green" if val2 > val1 else "red"
                weight2 = "bold" if val2 > val1 else "normal"
                
                # Simbolo differenza
                if diff > 0: diff_str = f"+{int(diff)}"
                elif diff < 0: diff_str = f"{int(diff)}" # il meno c'è già
                else: diff_str = "="
                
                # Rendering Riga
                rc1, rc2, rc3 = st.columns([1, 0.5, 1])
                
                with rc1:
                    st.markdown(f"<div style='text-align: center; color: {color1}; font-weight: {weight1}; font-size: 18px;'>{int(val1)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 14px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                
                with rc2:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 10px;'>{diff_str}</div>", unsafe_allow_html=True)
                    
                with rc3:
                    st.markdown(f"<div style='text-align: center; color: {color2}; font-weight: {weight2}; font-size: 18px;'>{int(val2)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 14px; color: gray;'>{label}</div>", unsafe_allow_html=True)
                
                st.divider()

            # --- RADAR CHART FINALE ---
            st.subheader("Radar Comparison")
            radar_cats = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            
            fig = go.Figure()
            
            # P1 Trace
            vals1 = [p1_data[c] for c in radar_cats]
            vals1 += [vals1[0]]
            fig.add_trace(go.Scatterpolar(r=vals1, theta=radar_cats + [radar_cats[0]], fill='toself', name=p1_data['short_name']))
            
            # P2 Trace
            vals2 = [p2_data[c] for c in radar_cats]
            vals2 += [vals2[0]]
            fig.add_trace(go.Scatterpolar(r=vals2, theta=radar_cats + [radar_cats[0]], fill='toself', name=p2_data['short_name']))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please select both players to compare.")

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
    
    st.divider()
    col_mod, _ = st.columns([1, 2])
    sel_formation = col_mod.selectbox("Select Tactical Module", list(FORMATIONS.keys()), index=0)
    
    if st.button("Show Squad"):
        with st.spinner(f"Scouting {sel_club_tac} ({sel_year_tac}) - Module {sel_formation}..."):
            df_squad = df_players.filter((col("club_name") == sel_club_tac) & (col("fifa_version") == sel_year_tac))
            cols_needed = ["short_name", "player_positions", "overall", "potential", "age", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
            cols_needed = [c for c in cols_needed if c in df_players.columns]
            squad_pdf = df_squad.select(cols_needed).orderBy(desc("overall")).toPandas()
            
            if not squad_pdf.empty:
                starters, bench = get_best_lineup(squad_pdf, module_name=sel_formation)
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
