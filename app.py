"""
Racial/Ethnic Disparities in Chronic Disease Surveillance
BIME 533 Final Project — Yang Yi, University of Washington
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, io

st.set_page_config(
    page_title="Chronic Disease Disparities | BRFSS 2019–2023",
    page_icon="📊", layout="wide", initial_sidebar_state="collapsed",
)

# ── CSS  ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base font scale — everything 2× larger */
  html, body, [class*="css"] { font-size: 20px !important; }

  .hero {
    background: linear-gradient(135deg,#1a3a5c 0%,#2e6da4 100%);
    color:white; padding:2.2rem 2.8rem; border-radius:14px; margin-bottom:1.5rem;
  }
  .hero h1 { font-size:2.4rem; margin:0 0 0.5rem 0; font-weight:800; }
  .hero p  { font-size:1.15rem; margin:0; opacity:0.88; line-height:1.6; }

  /* Metric cards */
  .cards { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.4rem; }
  .card  {
    background:#f0f5fb; border:1px solid #cdd8ea; border-radius:12px;
    padding:1.1rem 1.6rem; flex:1; min-width:150px; text-align:center;
  }
  .card .val { font-size:2.2rem; font-weight:800; color:#1a3a5c; line-height:1.1; }
  .card .lbl { font-size:0.85rem; color:#555; margin-top:0.35rem; line-height:1.4; }

  /* Callouts */
  .cw  { background:#fff8e1; border-left:5px solid #f4a300;
         padding:1rem 1.2rem; border-radius:6px; margin:.8rem 0; font-size:1rem; }
  .cg  { background:#e8f5e9; border-left:5px solid #2e7d32;
         padding:1rem 1.2rem; border-radius:6px; margin:.8rem 0; font-size:1rem; }
  .cb  { background:#e3f2fd; border-left:5px solid #1565c0;
         padding:1rem 1.2rem; border-radius:6px; margin:.8rem 0; font-size:1rem; }
  .cr  { background:#fce4ec; border-left:5px solid #c62828;
         padding:1rem 1.2rem; border-radius:6px; margin:.8rem 0; font-size:1rem; }

  /* Section headers */
  .sh { font-size:1.5rem; font-weight:800; color:#1a3a5c;
        border-bottom:3px solid #2e6da4; padding-bottom:0.4rem; margin-bottom:1.1rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] {
    font-size:1.05rem !important; font-weight:700;
    padding:0.6rem 1.4rem; border-radius:8px 8px 0 0;
  }

  /* Sidebar */
  .css-1d391kg { font-size:1rem; }

  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR)
YEARS    = [2019, 2020, 2021, 2022, 2023]

NO_HYP_YEARS = {2020, 2022}   # years with no hypertension variable available

RACE_ORDER = [
    "White non-Hispanic", "Black non-Hispanic", "Hispanic",
    "Asian non-Hispanic", "AIAN non-Hispanic",
    "NHPI non-Hispanic", "Multiracial non-Hispanic", "Other non-Hispanic",
]
RACE_LABEL_MAP = {
    "White NH":"White non-Hispanic","Black NH":"Black non-Hispanic",
    "Hispanic":"Hispanic","Asian NH":"Asian non-Hispanic",
    "AIAN NH":"AIAN non-Hispanic","NHPI NH":"NHPI non-Hispanic",
    "Multiracial NH":"Multiracial non-Hispanic","Other NH":"Other non-Hispanic",
}
RACE_COLORS = {
    "White non-Hispanic":"#4e79a7","Black non-Hispanic":"#e15759",
    "Hispanic":"#f28e2b","Asian non-Hispanic":"#59a14f",
    "AIAN non-Hispanic":"#b07aa1","NHPI non-Hispanic":"#76b7b2",
    "Multiracial non-Hispanic":"#ff9da7","Other non-Hispanic":"#9c755f",
}
INCOME_ORDER = ["Low (<$25k)","Middle ($25–50k)","Upper-middle ($50–75k)","High (>$75k)"]
EDUCA_ORDER  = [
    "Never attended school","Elementary (Grades 1–8)","Some high school",
    "High school graduate","Some college/tech school","College graduate",
]
STATE_ABBREV = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","DC":"DC","Florida":"FL",
    "Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN",
    "Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME",
    "Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}
# Healthy People 2030 national diabetes target
HP2030_DIABETES = 10.6

def fix_year_axis(fig):
    fig.update_xaxes(tickmode="array", tickvals=YEARS, tickformat="d")
    return fig

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def remap(df):
        df["race_label"] = df["race_label"].map(RACE_LABEL_MAP).fillna(df["race_label"])
        return df

    diab_race   = remap(pd.read_csv(os.path.join(OUT_DIR,"diabetes_by_race_year.csv")))
    hyp_race    = remap(pd.read_csv(os.path.join(OUT_DIR,"hypertension_by_race_year.csv")))
    ratios      = pd.read_csv(os.path.join(OUT_DIR,"disparity_ratios.csv"))
    ratios["comparison_group"] = ratios["comparison_group"].map(RACE_LABEL_MAP).fillna(ratios["comparison_group"])
    state_diab  = pd.read_csv(os.path.join(OUT_DIR,"diabetes_by_state_year.csv"))
    missingness = pd.read_csv(os.path.join(OUT_DIR,"race_missingness_by_state_year.csv"))
    income_race = remap(pd.read_csv(os.path.join(OUT_DIR,"diabetes_income_race.csv")))
    educa_race  = remap(pd.read_csv(os.path.join(OUT_DIR,"diabetes_education_race.csv")))
    return diab_race, hyp_race, ratios, state_diab, missingness, income_race, educa_race

diab_race, hyp_race, ratios, state_diab, missingness, income_race, educa_race = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Select Your Role")
    stakeholder = st.radio("", ["Epidemiologist","Community Health Worker"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ℹ️ About EPHS")
    st.markdown(
        "The **10 Essential Public Health Services** define what public health systems must do. "
        "This dashboard addresses:\n\n"
        "**EPHS #1 — Assess & Monitor**  \nTrack population health and health hazards over time\n\n"
        "**EPHS #3 — Communicate**  \nInform and educate the public, partners, and decision-makers"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;color:#888;'>"
        "BRFSS 2019–2023 · N=2,137,374 adults  \n"
        "Survey-weighted estimates (_LLCPWT)  \n"
        "Yang Yi · BIME 533 · UW · 2026"
        "</div>", unsafe_allow_html=True
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📊 Racial/Ethnic Disparities in Chronic Disease Surveillance</h1>
  <p>Using BRFSS 2019–2023 to assess how surveillance system design shapes the visibility
     of health inequities in diabetes and hypertension across demographic and geographic groups in the U.S.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Prevalence by Race",
    "📈 Disparity Trends",
    "🗺️ State Map",
    "🔍 Surveillance Quality",
    "💰 Income × Race",
    "🎓 Education × Race",
    "📋 Methods & Notes",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prevalence by Race
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sh">Chronic Disease Prevalence by Race/Ethnicity</div>', unsafe_allow_html=True)
    col_ctrl, _ = st.columns([1,3])
    with col_ctrl:
        yr1 = st.selectbox("Year", YEARS, index=4, key="yr1")

    df_d = diab_race[diab_race.year==yr1].copy()
    df_d = df_d[df_d.race_label.isin(RACE_ORDER)].sort_values("diabetes_prevalence_pct")
    df_h = hyp_race[hyp_race.year==yr1].copy()
    df_h = df_h[df_h.race_label.isin(RACE_ORDER)].sort_values("hyp_prevalence_pct")

    white_d = df_d[df_d.race_label=="White non-Hispanic"]["diabetes_prevalence_pct"].values
    black_d = df_d[df_d.race_label=="Black non-Hispanic"]["diabetes_prevalence_pct"].values
    black_h = df_h[df_h.race_label=="Black non-Hispanic"]["hyp_prevalence_pct"].values
    white_h = df_h[df_h.race_label=="White non-Hispanic"]["hyp_prevalence_pct"].values

    if len(white_d) and len(black_d):
        ratio_bd = black_d[0] / white_d[0]
        bh_val = f"{black_h[0]:.1f}%" if len(black_h) else "N/A"
        wh_val = f"{white_h[0]:.1f}%" if len(white_h) else "N/A"
        st.markdown(f"""
        <div class="cards">
          <div class="card"><div class="val">{black_d[0]:.1f}%</div><div class="lbl">Black non-Hispanic<br>Diabetes ({yr1})</div></div>
          <div class="card"><div class="val">{white_d[0]:.1f}%</div><div class="lbl">White non-Hispanic<br>Diabetes ({yr1})</div></div>
          <div class="card"><div class="val">{ratio_bd:.2f}×</div><div class="lbl">Black/White<br>Disparity Ratio</div></div>
          <div class="card"><div class="val">{bh_val}</div><div class="lbl">Black non-Hispanic<br>Hypertension ({yr1})</div></div>
          <div class="card"><div class="val">{wh_val}</div><div class="lbl">White non-Hispanic<br>Hypertension ({yr1})</div></div>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Diabetes Prevalence (%) — {yr1}**")
        fig = px.bar(df_d, x="diabetes_prevalence_pct", y="race_label", orientation="h",
                     color="race_label", color_discrete_map=RACE_COLORS,
                     labels={"diabetes_prevalence_pct":"Prevalence (%)","race_label":""},
                     text="diabetes_prevalence_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_vline(x=HP2030_DIABETES, line_dash="dash", line_color="#c00",
                      annotation_text=f"HP2030 target ({HP2030_DIABETES}%)", annotation_position="top right")
        fig.update_layout(showlegend=False, height=380,
                          margin=dict(l=10,r=80,t=10,b=30),
                          xaxis=dict(range=[0, df_d.diabetes_prevalence_pct.max()*1.25]),
                          font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True, key="bar_diab")

        if stakeholder == "Epidemiologist":
            st.markdown(f"<div class='cb'>Black non-Hispanic adults have a <b>{ratio_bd:.2f}× higher</b> weighted diabetes prevalence than White non-Hispanic adults ({black_d[0]:.1f}% vs {white_d[0]:.1f}%). The red dashed line marks the Healthy People 2030 national target of {HP2030_DIABETES}% — most minority groups remain above this target.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='cb'>📌 About <b>{int(round(black_d[0]))} out of every 100</b> Black non-Hispanic adults have diabetes — compared to <b>{int(round(white_d[0]))} out of 100</b> White non-Hispanic adults. The national goal is to get below {HP2030_DIABETES}%.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"**Hypertension Prevalence (%) — {yr1}**")
        if yr1 in NO_HYP_YEARS:
            st.info(f"⚠️ Hypertension data is not available for {yr1} — the CDC did not release a computed hypertension variable in the {yr1} BRFSS public file.")
        else:
            fig2 = px.bar(df_h, x="hyp_prevalence_pct", y="race_label", orientation="h",
                          color="race_label", color_discrete_map=RACE_COLORS,
                          labels={"hyp_prevalence_pct":"Prevalence (%)","race_label":""},
                          text="hyp_prevalence_pct")
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.update_layout(showlegend=False, height=380,
                               margin=dict(l=10,r=80,t=10,b=30),
                               xaxis=dict(range=[0, df_h.hyp_prevalence_pct.max()*1.25]),
                               font=dict(size=15))
            st.plotly_chart(fig2, use_container_width=True, key="bar_hyp")
            hisp_h = df_h[df_h.race_label=="Hispanic"]["hyp_prevalence_pct"].values
            hisp_val = f"{hisp_h[0]:.1f}%" if len(hisp_h) else "N/A"
            if len(black_h) and len(white_h):
                if stakeholder == "Epidemiologist":
                    st.markdown(f"<div class='cb'>Black non-Hispanic hypertension ({black_h[0]:.1f}%) substantially exceeds White non-Hispanic ({white_h[0]:.1f}%). Hispanic adults show a paradoxically lower rate ({hisp_val}) despite socioeconomic disadvantage — the Hispanic epidemiological paradox.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='cb'>📌 <b>{black_h[0]:.0f} out of 100</b> Black non-Hispanic adults have high blood pressure — the highest of any group. Blood pressure screening should be a priority for Black community outreach programs.</div>", unsafe_allow_html=True)

    if stakeholder == "Epidemiologist":
        with st.expander("📋 View full prevalence table"):
            merged = df_d[["race_label","diabetes_prevalence_pct","n"]].merge(
                df_h[["race_label","hyp_prevalence_pct"]], on="race_label", how="outer")
            merged.columns = ["Race/Ethnicity","Diabetes %","N","Hypertension %"]
            st.dataframe(merged.sort_values("Diabetes %",ascending=False).reset_index(drop=True))
        # Download button
        csv = merged.to_csv(index=False).encode()
        st.download_button("⬇️ Download prevalence table (CSV)", csv,
                           f"prevalence_by_race_{yr1}.csv", "text/csv", key="dl_tab1")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Disparity Trends
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sh">Racial/Ethnic Disparity Trends, 2019–2023</div>', unsafe_allow_html=True)
    st.markdown("**Disparity ratio** = minority group prevalence ÷ White non-Hispanic prevalence. Ratio = **1.0** means no disparity.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Diabetes Disparity Ratio**")
        fig_r = px.line(ratios.dropna(subset=["diabetes_ratio"]),
                        x="year", y="diabetes_ratio", color="comparison_group", markers=True,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"diabetes_ratio":"Prevalence Ratio","year":"Year","comparison_group":"Group"})
        fig_r.add_hline(y=1.0, line_dash="dash", line_color="#888", annotation_text="No disparity (1.0)")
        fix_year_axis(fig_r)
        fig_r.update_layout(height=360, margin=dict(t=10,b=20), font=dict(size=14))
        st.plotly_chart(fig_r, use_container_width=True, key="rat_diab")

    with col2:
        st.markdown("**Hypertension Disparity Ratio**")
        fig_rh = px.line(ratios.dropna(subset=["hypertension_ratio"]),
                         x="year", y="hypertension_ratio", color="comparison_group", markers=True,
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         labels={"hypertension_ratio":"Prevalence Ratio","year":"Year","comparison_group":"Group"})
        fig_rh.add_hline(y=1.0, line_dash="dash", line_color="#888", annotation_text="No disparity (1.0)")
        fix_year_axis(fig_rh)
        fig_rh.update_layout(height=360, margin=dict(t=10,b=20), font=dict(size=14))
        st.plotly_chart(fig_rh, use_container_width=True, key="rat_hyp")

    st.markdown("---")
    st.markdown("**Absolute Prevalence Over Time**")
    c1, c2 = st.columns([1,2])
    with c1:
        condition = st.radio("Condition", ["Diabetes","Hypertension"], horizontal=True, key="t2cond")
    with c2:
        src = diab_race if condition=="Diabetes" else hyp_race
        sel_races = st.multiselect("Race/ethnicity groups",
            [r for r in RACE_ORDER if r in src.race_label.unique()],
            default=["White non-Hispanic","Black non-Hispanic","Hispanic","Asian non-Hispanic"],
            key="t2races")

    df_abs = src[src.race_label.isin(sel_races)]
    y_col = "diabetes_prevalence_pct" if condition=="Diabetes" else "hyp_prevalence_pct"
    y_lbl = f"{condition} Prevalence (%)"
    fig_a = px.line(df_abs, x="year", y=y_col, color="race_label", markers=True,
                    color_discrete_map=RACE_COLORS,
                    labels={y_col:y_lbl,"year":"Year","race_label":"Race/Ethnicity"})
    if condition == "Diabetes":
        fig_a.add_hline(y=HP2030_DIABETES, line_dash="dot", line_color="#c00",
                        annotation_text=f"HP2030 target {HP2030_DIABETES}%", annotation_position="bottom right")
    fix_year_axis(fig_a)
    fig_a.update_layout(height=400, margin=dict(t=10), font=dict(size=14))
    st.plotly_chart(fig_a, use_container_width=True, key="abs_trend")

    if stakeholder == "Epidemiologist":
        st.markdown("<div class='cw'>⚠️ <b>Data notes:</b> 2020 and 2022 hypertension data are unavailable (no computed variable in CDC public files). 2020 hypertension gap in disparity chart reflects missing data, not a true absence of disparity. Trends should be interpreted with these gaps in mind.</div>", unsafe_allow_html=True)
        st.markdown("<div class='cg'>📊 <b>EPHS #1:</b> Persistent year-over-year disparity ratios for Black non-Hispanic adults (~1.4× for diabetes, ~1.2× for hypertension) demonstrate durable structural inequities. Surveillance systems reporting only overall prevalence would mask these patterns entirely.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='cb'>📌 <b>Key message:</b> The gap between Black non-Hispanic and White non-Hispanic communities for diabetes has not closed in five years. Targeted community programs — including culturally tailored diabetes prevention, grocery access, and clinical partnerships — are urgently needed.</div>", unsafe_allow_html=True)

    # Download
    dl_df = df_abs[["year","race_label",y_col]].copy()
    st.download_button("⬇️ Download trend data (CSV)",
                       dl_df.to_csv(index=False).encode(),
                       f"{condition.lower()}_trend_by_race.csv","text/csv", key="dl_tab2")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — State Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sh">State-Level Diabetes Prevalence</div>', unsafe_allow_html=True)
    col_ctrl, _ = st.columns([1,3])
    with col_ctrl:
        yr3 = st.selectbox("Year", YEARS, index=4, key="yr3")

    df_map = state_diab[state_diab.year==yr3].copy()
    df_map = df_map[~df_map.state.isin(["Puerto Rico","Guam","Virgin Islands"])]
    df_map["abbrev"] = df_map["state"].map(STATE_ABBREV)
    df_map = df_map.dropna(subset=["abbrev"])

    fig_map = px.choropleth(
        df_map, locations="abbrev", locationmode="USA-states",
        color="diabetes_prevalence_pct", scope="usa",
        color_continuous_scale="YlOrRd",
        labels={"diabetes_prevalence_pct":"Diabetes (%)"},
        hover_name="state",
        hover_data={"abbrev":False,"diabetes_prevalence_pct":":.1f"},
    )
    fig_map.update_layout(
        geo=dict(showlakes=False),
        coloraxis_colorbar=dict(title="Prevalence (%)"),
        height=520, margin=dict(t=10,b=10,l=0,r=0), font=dict(size=14),
    )
    st.plotly_chart(fig_map, use_container_width=True, key="state_map")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 — Highest Prevalence**")
        st.dataframe(df_map.sort_values("diabetes_prevalence_pct",ascending=False).head(10)
                     [["state","diabetes_prevalence_pct"]].rename(
                         columns={"state":"State","diabetes_prevalence_pct":"Diabetes %"})
                     .reset_index(drop=True), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Bottom 10 — Lowest Prevalence**")
        st.dataframe(df_map.sort_values("diabetes_prevalence_pct").head(10)
                     [["state","diabetes_prevalence_pct"]].rename(
                         columns={"state":"State","diabetes_prevalence_pct":"Diabetes %"})
                     .reset_index(drop=True), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Compare Your State to the National Average**")
    nat_avg = round(df_map["diabetes_prevalence_pct"].mean(), 2)
    sel_state = st.selectbox("Select a state", sorted(df_map["state"].dropna().tolist()), key="state_sel")
    state_val = df_map[df_map.state==sel_state]["diabetes_prevalence_pct"].values
    if len(state_val):
        diff = round(state_val[0] - nat_avg, 2)
        direction = "above" if diff > 0 else "below"
        color_class = "cr" if diff > 0 else "cg"
        st.markdown(f"<div class='{color_class}'><b>{sel_state}</b>: {state_val[0]:.1f}% diabetes prevalence — "
                    f"<b>{abs(diff):.1f} percentage points {direction}</b> the national average ({nat_avg}%).</div>",
                    unsafe_allow_html=True)

    if stakeholder == "Epidemiologist":
        st.markdown("<div class='cg'>📊 <b>EPHS #1:</b> Southern states consistently rank highest in diabetes prevalence, reflecting structural determinants including poverty concentration, racial residential segregation, and healthcare access gaps. State-level rates without race stratification confound true disease burden with population composition differences.</div>", unsafe_allow_html=True)
    else:
        top = df_map.sort_values("diabetes_prevalence_pct",ascending=False).iloc[0]
        bot = df_map.sort_values("diabetes_prevalence_pct").iloc[0]
        st.markdown(f"<div class='cb'>📌 <b>{top.state}</b> has the highest diabetes rate ({top.diabetes_prevalence_pct:.1f}%), and <b>{bot.state}</b> has the lowest ({bot.diabetes_prevalence_pct:.1f}%). If you work in a high-burden state, connecting residents to CDC's National Diabetes Prevention Program (National DPP) can reduce risk by up to 58%.</div>", unsafe_allow_html=True)

    st.download_button("⬇️ Download state data (CSV)",
                       df_map[["state","diabetes_prevalence_pct","n"]].to_csv(index=False).encode(),
                       f"state_diabetes_{yr3}.csv","text/csv", key="dl_tab3")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Surveillance Quality
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sh">Surveillance Quality: Race/Ethnicity Data Missingness</div>', unsafe_allow_html=True)

    if stakeholder == "Epidemiologist":
        st.markdown("When a respondent's race/ethnicity is coded as 'Don't know/Refused' (_RACE = 9), their record is **excluded from all race-stratified prevalence analyses**. High missingness rates systematically suppress disparity estimates — particularly in states where minority populations are most at risk.")
    else:
        st.markdown("If a survey doesn't record someone's race or ethnicity, that person becomes **invisible in health reports**. States with high missing race data can't tell you which communities need the most help.")

    col_ctrl, _ = st.columns([1,3])
    with col_ctrl:
        yr4 = st.selectbox("Year", YEARS, index=4, key="yr4")

    df_my = missingness[missingness.year==yr4].copy()
    df_my = df_my[~df_my.state.isin(["Puerto Rico","Guam","Virgin Islands"])]
    df_my["abbrev"] = df_my["state"].map(STATE_ABBREV)
    df_my = df_my.dropna(subset=["abbrev"])

    c1, c2 = st.columns([2,1])
    with c1:
        fig_mm = px.choropleth(
            df_my, locations="abbrev", locationmode="USA-states",
            color="race_missing_pct", scope="usa", color_continuous_scale="Reds",
            labels={"race_missing_pct":"Missing (%)"},
            hover_name="state",
            hover_data={"abbrev":False,"race_missing_pct":":.2f","total_respondents":True},
        )
        fig_mm.update_layout(coloraxis_colorbar=dict(title="Missing (%)"),
                             height=440, margin=dict(t=10,b=10,l=0,r=0), font=dict(size=14))
        st.plotly_chart(fig_mm, use_container_width=True, key="miss_map")
    with c2:
        st.markdown(f"**States with Most Missing Race Data ({yr4})**")
        st.dataframe(df_my.sort_values("race_missing_pct",ascending=False).head(10)
                     [["state","race_missing_pct","total_respondents"]].rename(
                         columns={"state":"State","race_missing_pct":"Missing (%)","total_respondents":"N"})
                     .reset_index(drop=True), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Missingness Trend Over Time**")
    all_states = sorted(missingness.state.dropna().unique())
    sel_st = st.multiselect("Select states", all_states,
        default=["Mississippi","Texas","New York","California","Washington"], key="miss_st")
    fig_mt = px.line(missingness[missingness.state.isin(sel_st)],
                     x="year", y="race_missing_pct", color="state", markers=True,
                     labels={"race_missing_pct":"Race Missing (%)","year":"Year","state":"State"})
    fix_year_axis(fig_mt)
    fig_mt.update_layout(height=340, margin=dict(t=10), font=dict(size=14))
    st.plotly_chart(fig_mt, use_container_width=True, key="miss_trend")

    worst = df_my.sort_values("race_missing_pct",ascending=False).iloc[0]
    if stakeholder == "Epidemiologist":
        st.markdown(f"<div class='cw'>⚠️ <b>Surveillance Limitation:</b> {worst.state} has {worst.race_missing_pct:.1f}% missing race data in {yr4} (~1 in {int(round(100/worst.race_missing_pct))} respondents). This missingness is likely non-random — respondents declining race questions may disproportionately belong to minority groups with lower institutional trust, creating systematic underestimation of disparities where they are greatest.</div>", unsafe_allow_html=True)
        st.markdown("<div class='cg'>📊 <b>EPHS #1:</b> Race/ethnicity missingness should be reported as a mandatory data quality metric alongside prevalence estimates in surveillance reports. Without this transparency, decision-makers cannot assess the reliability of race-stratified estimates.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='cw'>⚠️ In {worst.state}, about <b>1 in {int(round(100/worst.race_missing_pct))} survey respondents</b> did not have their race recorded in {yr4}. This means health reports from this state can't accurately identify which communities are most affected by chronic disease.</div>", unsafe_allow_html=True)
        st.markdown("<div class='cb'>📢 <b>EPHS #3 — Communicate:</b> When race data is missing, health agencies can't send targeted health information back to the communities who need it most. Community health workers can help by encouraging survey participation and explaining why race data matters for health equity.</div>", unsafe_allow_html=True)

    st.download_button("⬇️ Download missingness data (CSV)",
                       df_my[["state","race_missing_pct","total_respondents","race_missing_n"]]
                       .to_csv(index=False).encode(),
                       f"missingness_{yr4}.csv","text/csv", key="dl_tab4")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Income × Race
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sh">Diabetes Prevalence by Household Income and Race/Ethnicity</div>', unsafe_allow_html=True)
    st.markdown("Income and race **independently and jointly** shape chronic disease risk. "
                "This panel shows whether the racial gap in diabetes persists even after accounting for income level.")

    c1, c2 = st.columns([1,2])
    with c1:
        yr5 = st.selectbox("Year", YEARS, index=4, key="yr5")
    with c2:
        sel_r5 = st.multiselect("Race/ethnicity groups", RACE_ORDER,
            default=["White non-Hispanic","Black non-Hispanic","Hispanic","Asian non-Hispanic"], key="r5")

    df_i = income_race[(income_race.year==yr5)&(income_race.race_label.isin(sel_r5))].copy()
    df_i["income_group"] = pd.Categorical(df_i.income_group, categories=INCOME_ORDER, ordered=True)
    df_i = df_i.sort_values("income_group")

    if df_i.empty:
        st.warning("No data for this selection.")
    else:
        c1, c2 = st.columns([3,2])
        with c1:
            fig_il = px.line(df_i, x="income_group", y="diabetes_prevalence_pct",
                             color="race_label", markers=True, color_discrete_map=RACE_COLORS,
                             labels={"diabetes_prevalence_pct":"Diabetes Prevalence (%)","income_group":"Household Income","race_label":"Race/Ethnicity"})
            fig_il.update_layout(height=420, margin=dict(t=10), font=dict(size=14),
                                 xaxis=dict(categoryorder="array",categoryarray=INCOME_ORDER))
            st.plotly_chart(fig_il, use_container_width=True, key="inc_line")
        with c2:
            st.markdown(f"**Key Findings — {yr5}**")
            st.markdown("""
**Income gradient is universal** — diabetes risk falls as income rises for all racial groups.

**Racial gaps persist within income levels** — even among high-income adults, Black non-Hispanic rates exceed White non-Hispanic rates at the same income level.

**Race captures more than income** — neighborhood environment, healthcare access barriers, chronic stress, and historical discrimination contribute to gaps not explained by income alone.

**Surveillance implication** — reporting race and income separately misses this compounding effect; cross-tabulation is needed.
            """)

        fig_ib = px.bar(df_i, x="race_label", y="diabetes_prevalence_pct", color="income_group",
                        barmode="group",
                        color_discrete_sequence=["#d73027","#fc8d59","#fee090","#91cf60"],
                        category_orders={"income_group":INCOME_ORDER},
                        labels={"diabetes_prevalence_pct":"Diabetes (%)","race_label":"Race/Ethnicity","income_group":"Income Group"})
        fig_ib.update_layout(height=380, margin=dict(t=10), font=dict(size=14),
                             legend=dict(title="Income Group",orientation="h",y=-0.25))
        st.plotly_chart(fig_ib, use_container_width=True, key="inc_bar")

        if stakeholder == "Epidemiologist":
            st.markdown("<div class='cg'>📊 <b>EPHS #1:</b> Single-variable stratification by race or income alone understates burden for multiply-disadvantaged populations. Surveillance reports should standardly publish race × income cross-tabulations to support equity-focused resource allocation.</div>", unsafe_allow_html=True)
        else:
            lb = df_i[(df_i.race_label=="Black non-Hispanic")&(df_i.income_group=="Low (<$25k)")]["diabetes_prevalence_pct"].values
            lw = df_i[(df_i.race_label=="White non-Hispanic")&(df_i.income_group=="Low (<$25k)")]["diabetes_prevalence_pct"].values
            if len(lb) and len(lw):
                st.markdown(f"<div class='cb'>📌 Among low-income adults, <b>{lb[0]:.1f}%</b> of Black non-Hispanic vs <b>{lw[0]:.1f}%</b> of White non-Hispanic adults have diabetes. Outreach programs targeting low-income Black communities will reach the highest-burden population.</div>", unsafe_allow_html=True)

        st.download_button("⬇️ Download income × race data (CSV)",
                           df_i[["race_label","income_group","diabetes_prevalence_pct","n"]]
                           .to_csv(index=False).encode(),
                           f"income_race_diabetes_{yr5}.csv","text/csv", key="dl_tab5")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Education × Race (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sh">Diabetes Prevalence by Education Level and Race/Ethnicity</div>', unsafe_allow_html=True)
    st.markdown("Education is a powerful social determinant of health, often more strongly linked to chronic disease than income. "
                "This panel examines whether racial disparities in diabetes persist across all education levels.")

    c1, c2 = st.columns([1,2])
    with c1:
        yr6 = st.selectbox("Year", YEARS, index=4, key="yr6")
    with c2:
        sel_r6 = st.multiselect("Race/ethnicity groups", RACE_ORDER,
            default=["White non-Hispanic","Black non-Hispanic","Hispanic","Asian non-Hispanic"], key="r6")

    df_e = educa_race[(educa_race.year==yr6)&(educa_race.race_label.isin(sel_r6))].copy()
    df_e["educa_label"] = pd.Categorical(df_e.educa_label, categories=EDUCA_ORDER, ordered=True)
    df_e = df_e.sort_values("educa_label")

    if df_e.empty:
        st.warning("No data for this selection.")
    else:
        fig_el = px.line(df_e, x="educa_label", y="diabetes_prevalence_pct",
                         color="race_label", markers=True, color_discrete_map=RACE_COLORS,
                         labels={"diabetes_prevalence_pct":"Diabetes Prevalence (%)","educa_label":"Education Level","race_label":"Race/Ethnicity"})
        fig_el.update_layout(height=440, margin=dict(t=10), font=dict(size=14),
                             xaxis=dict(categoryorder="array",categoryarray=EDUCA_ORDER,
                                        tickangle=-25))
        st.plotly_chart(fig_el, use_container_width=True, key="edu_line")

        fig_eb = px.bar(df_e, x="race_label", y="diabetes_prevalence_pct", color="educa_label",
                        barmode="group",
                        color_discrete_sequence=px.colors.sequential.Blues[1:],
                        category_orders={"educa_label":EDUCA_ORDER},
                        labels={"diabetes_prevalence_pct":"Diabetes (%)","race_label":"Race/Ethnicity","educa_label":"Education Level"})
        fig_eb.update_layout(height=380, margin=dict(t=10), font=dict(size=14),
                             legend=dict(title="Education",orientation="h",y=-0.3))
        st.plotly_chart(fig_eb, use_container_width=True, key="edu_bar")

        if stakeholder == "Epidemiologist":
            st.markdown("<div class='cb'>Education exhibits a strong inverse gradient with diabetes risk for all groups — higher education is strongly protective. However, racial gaps persist within each education category, particularly at the college graduate level, suggesting race-specific barriers beyond educational attainment (e.g., occupational segregation, neighborhood-level food environment).</div>", unsafe_allow_html=True)
            st.markdown("<div class='cg'>📊 <b>EPHS #1:</b> Cross-tabulated surveillance by race × education enables identification of college-educated Black adults as a high-burden subgroup often overlooked by income-only stratification — enabling more precise targeting of prevention resources.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='cb'>📌 Even Black and Hispanic adults who graduated college still have higher diabetes rates than White adults at the same education level. This tells us that education alone doesn't close the health gap — systemic barriers beyond schooling must also be addressed by community programs.</div>", unsafe_allow_html=True)

        st.download_button("⬇️ Download education × race data (CSV)",
                           df_e[["race_label","educa_label","diabetes_prevalence_pct","n"]]
                           .to_csv(index=False).encode(),
                           f"education_race_diabetes_{yr6}.csv","text/csv", key="dl_tab6")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Methods & Notes
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="sh">Methods, Data Notes, and Limitations</div>', unsafe_allow_html=True)

    with st.expander("📦 Data Sources", expanded=True):
        st.markdown("""
| Dataset | Source | Years | Variables Used |
|---|---|---|---|
| **BRFSS** (Behavioral Risk Factor Surveillance System) | CDC / NCHS | 2019–2023 | Diabetes (DIABETE4), hypertension (_RFHYPE5/_RFHYPE6), race (_RACE/_RACE1), income, education, survey weights |
| **Healthy People 2030 targets** | ODPHP | 2030 | Diabetes target: 10.6% population prevalence |

**Total respondents:** 2,137,374 U.S. adults (18+), non-institutionalized
**Survey design:** Complex multistage probability sample; all estimates use BRFSS finalized annual weights (_LLCPWT) to represent the U.S. adult population.
        """)

    with st.expander("📐 Analytical Methods"):
        st.markdown("""
**Survey-weighted prevalence:**
All prevalence estimates are computed as weighted proportions:

`Prevalence = Σ(outcome × weight) / Σ(weight)`

where `outcome` is a binary 0/1 indicator and `weight` is the BRFSS finalized annual weight (`_LLCPWT`).

**Disparity ratio:** Prevalence ratio of a minority group relative to White non-Hispanic adults in the same year.

**Income harmonization:** INCOME2 (2019–2020, 8 categories) and INCOME3 (2021–2023, 11 categories) were harmonized into 4 comparable groups:
- Low (<$25k): codes 1–4 in both versions
- Middle ($25–50k): codes 5–6
- Upper-middle ($50–75k): code 7
- High (>$75k): code 8 (INCOME2) / codes 8–11 (INCOME3)

**Race/ethnicity coding:** BRFSS computed variable `_RACE` (codes 1–8). Code 9 (Don't know/Refused) is excluded from prevalence analysis but counted in the missingness panel.
        """)

    with st.expander("⚠️ Known Data Limitations"):
        st.markdown("""
| Limitation | Years Affected | Notes |
|---|---|---|
| Hypertension variable unavailable | 2020, 2022 | CDC did not publish computed hypertension variable in public files |
| Race variable renamed | 2022 | `_RACE` renamed to `_RACE1`; recoded to `_RACE` for consistency |
| Self-reported diabetes | All years | BRFSS relies on self-report; clinical underdiagnosis may lead to underestimation, particularly in under-served populations |
| BRFSS excludes institutionalized adults | All years | Persons in nursing homes, correctional facilities, etc. are not sampled — groups with disproportionate minority representation |
| State-level sample sizes vary | All years | Small states (e.g., Wyoming, Vermont) have smaller samples, leading to less stable estimates for small racial subgroups |
| NHANES comparison not included | — | A full validation of BRFSS self-report against NHANES clinical measures (HbA1c) would strengthen surveillance quality conclusions |
        """)

    with st.expander("🎯 Alignment with 10 Essential Public Health Services"):
        st.markdown("""
| EPHS | How This Project Addresses It |
|---|---|
| **#1 — Assess and Monitor Population Health** | Primary focus: quantifying racial/ethnic disparities in diabetes and hypertension using national surveillance data; evaluating surveillance data quality (missingness) |
| **#3 — Communicate Effectively to Inform and Educate** | Dual stakeholder views (epidemiologist / CHW) translate technical findings into actionable guidance; interactive dashboard makes data accessible to diverse audiences |
        """)

    with st.expander("👥 Stakeholder Information Needs"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**State-Level Epidemiologists**
- Weighted prevalence estimates with methodological transparency
- Race × income × education cross-tabulations
- Data quality metrics (missingness rates) by state
- Disparity ratios over time to assess whether gaps are narrowing
- Benchmark against Healthy People 2030 national targets
            """)
        with col2:
            st.markdown("""
**Community Health Workers**
- Plain-language "1 in X" framing of disease burden
- Local (state-level) data for their specific communities
- Identification of highest-burden subgroups for screening prioritization
- Actionable links to evidence-based programs (e.g., National DPP)
- Clear communication of why survey participation and race data collection matter
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='color:#aaa;font-size:0.78rem;text-align:center;'>"
    "Data: CDC BRFSS 2019–2023. Survey-weighted using _LLCPWT. "
    "NH = Non-Hispanic. AIAN = American Indian/Alaska Native. NHPI = Native Hawaiian/Pacific Islander. "
    "Hypertension unavailable for 2020 and 2022. 2019 race variable: _RACE. 2022 race variable: _RACE1 (recoded). "
    "HP2030 = Healthy People 2030 national diabetes target (10.6%). "
    "Yang Yi · BIME 533 · University of Washington · 2026"
    "</div>", unsafe_allow_html=True
)
