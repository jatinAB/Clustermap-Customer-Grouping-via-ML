# app.py - Clustermap: Customer Grouping via ML (Business Insight Dashboard)
# Upgraded by Jatin Bandekar
# Features added: Business Insight Cards, Cluster Comparison tab, Auto Persona Generator

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------
# Page config / header
# -----------------------
st.set_page_config(page_title="Clustermap: Customer Grouping via ML", layout="wide")
st.title("🗺️ Clustermap: Customer Grouping via ML")
st.caption("Developed by **Jatin Bandekar** — Business Insight Dashboard")

# -----------------------
# Helper utilities
# -----------------------
@st.cache_data
def load_data():
    for name in ("ecommerce_customer_data_full.csv", "ecommerce_customer_data.csv"):
        try:
            df = pd.read_csv(name)
            st.session_state["_data_file"] = name
            return df
        except Exception:
            continue
    st.warning("⚠️ No dataset found. Please place 'ecommerce_customer_data_full.csv' or 'ecommerce_customer_data.csv' in the project folder.")
    return pd.DataFrame()

def ensure_columns(df):
    expected = ['Age','Recency','Frequency','Monetary',
                'Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend','Region','CustomerID','Gender']
    for c in expected:
        if c not in df.columns:
            if c == 'CustomerID':
                df['CustomerID'] = np.arange(1, len(df)+1)
            elif c == 'Region':
                df['Region'] = 'Unknown'
            elif c == 'Gender':
                df['Gender'] = 'Unknown'
            else:
                df[c] = 0
    return df

def compute_basic_churn_cltv(df):
    if 'churn_proba' not in df.columns:
        df['churn_proba'] = (df['Recency'] / (df['Recency'].max() + 1)).clip(0,1)
    if 'avg_order_value' not in df.columns:
        df['avg_order_value'] = df.apply(lambda r: r['Monetary'] / r['Frequency'] if r['Frequency']>0 else r['Monetary'], axis=1)
    if 'CLTV_simple' not in df.columns:
        df['retention_prob'] = 1 - df['churn_proba']
        df['exp_purchases_next_yr'] = df['Frequency']
        df['CLTV_simple'] = df['avg_order_value'] * df['exp_purchases_next_yr'] * df['retention_prob']
    return df

@st.cache_data
def train_kmeans(df, n_clusters=5):
    features = ['Age','Recency','Frequency','Monetary']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, X_scaled

def cluster_summary(df):
    # returns summary dataframe and sizes
    cols = ['Age','Recency','Frequency','Monetary','CLTV_simple','churn_proba']
    agg = df.groupby('Cluster')[cols].agg(['mean','median','count']).round(2)
    sizes = df['Cluster'].value_counts().sort_index()
    return agg, sizes

def generate_persona_text(cluster_id, df_cluster):
    # dynamic persona generation (simple rule-based)
    age_mean = df_cluster['Age'].mean()
    cltv_mean = df_cluster['CLTV_simple'].mean()
    freq_mean = df_cluster['Frequency'].mean()
    recency_mean = df_cluster['Recency'].mean()
    top_category = df_cluster[['Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend']].mean().idxmax()
    top_category_readable = top_category.replace('_Spend','').replace('_',' ').title()

    persona = f"**Cluster {cluster_id} Persona** — Avg Age: {age_mean:.0f}. "
    persona += f"Average CLTV ≈ ₹{cltv_mean:,.0f}. "
    persona += f"Purchasing frequency is about {freq_mean:.1f} transactions/year and average recency is {recency_mean:.0f} days. "
    persona += f"Top spend category: **{top_category_readable}**. "
    # simple recommendation rules
    if cltv_mean > df['CLTV_simple'].mean():
        persona += "This is a **high-value** segment — consider VIP programs and retention offers. "
    else:
        persona += "This segment has **lower CLTV** — focus on acquisition & upsell strategies. "

    if recency_mean > 180:
        persona += "Customers show low recent activity — consider re-engagement campaigns. "
    return persona

# -----------------------
# Load & prepare data
# -----------------------
df = load_data()
if df.empty:
    st.stop()

df = ensure_columns(df)
df = compute_basic_churn_cltv(df)

# Train or reuse clusters
if 'Cluster' not in df.columns:
    kmeans, scaler, X_scaled = train_kmeans(df, n_clusters=5)
    df['Cluster'] = kmeans.labels_
else:
    features = ['Age','Recency','Frequency','Monetary']
    scaler = StandardScaler()
    scaler.fit(df[features].fillna(0))

# PCA for visualization
features = ['Age','Recency','Frequency','Monetary']
pca = PCA(n_components=2, random_state=22)
pca_vals = pca.fit_transform(scaler.transform(df[features].fillna(0)))
df['PCA1'] = pca_vals[:,0]
df['PCA2'] = pca_vals[:,1]

# -----------------------
# Sidebar - controls
# -----------------------
st.sidebar.markdown("### 🎛️ Filters & Controls")
clusters_available = sorted(df['Cluster'].unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", clusters_available, default=clusters_available)
age_min, age_max = st.sidebar.slider("Age range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
region_opts = df['Region'].unique().tolist()
sel_regions = st.sidebar.multiselect("Regions", region_opts, default=region_opts)
cltv_min = float(df['CLTV_simple'].min())
cltv_max = float(df['CLTV_simple'].max())
sel_cltv = st.sidebar.slider("CLTV range", float(round(cltv_min,0)), float(round(cltv_max,0)), (float(round(cltv_min,0)), float(round(cltv_max,0))))

# filtered df for main visuals
df_filtered = df[
    (df['Cluster'].isin(sel_clusters)) &
    (df['Age'] >= age_min) & (df['Age'] <= age_max) &
    (df['Region'].isin(sel_regions)) &
    (df['CLTV_simple'] >= sel_cltv[0]) & (df['CLTV_simple'] <= sel_cltv[1])
].copy()

# download filtered data
st.sidebar.markdown("### Data Export")
st.sidebar.download_button(
    label="⬇️ Download filtered CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="filtered_customers.csv",
    mime="text/csv"
)

# -----------------------
# Tab layout
# -----------------------
tab_overview, tab_profiles, tab_prediction, tab_insights, tab_compare = st.tabs(
    ["📊 Overview", "👥 Customer Profiles", "🤖 Prediction", "📈 Insights", "🔍 Compare Clusters"]
)

# -----------------------
# Tab: Overview
# -----------------------
with tab_overview:
    st.subheader("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers (filtered)", len(df_filtered))
    c2.metric("Total Monetary (approx.)", f"₹{df_filtered['Monetary'].sum():,.0f}")
    c3.metric("Avg Frequency", f"{df_filtered['Frequency'].mean():.2f}")
    c4.metric("Avg CLTV", f"₹{df_filtered['CLTV_simple'].mean():,.0f}")

    st.markdown("#### PCA: Cluster separation (interactive)")
    fig = px.scatter(df_filtered, x='PCA1', y='PCA2', color='Cluster', hover_data=['CustomerID','Monetary','Frequency','CLTV_simple'], template='plotly_white', height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monetary vs Frequency")
    fig2 = px.scatter(df_filtered, x='Frequency', y='Monetary', color='Cluster', size='Monetary', hover_data=['CustomerID','Age'], template='plotly_white', height=420)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Tab: Customer Profiles
# -----------------------
with tab_profiles:
    st.subheader("Customer Profiles & Drill-down")

    # friendly cluster names and descriptions (you can update text)
    cluster_names = {
        0: ("Loyal Shoppers", "Frequent repeat customers with high CLTV."),
        1: ("High-Value Wanderers", "Occasional premium spenders."),
        2: ("Trend Seekers", "Young, dynamic, category-diverse spenders."),
        3: ("Value Maximizers", "Price-sensitive but regular buyers."),
        4: ("New Entrants", "Recently joined or low-engagement customers.")
    }

    cols = st.columns(5)
    for i, c in enumerate(sorted(df['Cluster'].unique())):
        name, desc = cluster_names.get(c, (f"Cluster {c}", ""))
        cols[i].markdown(f"**{c} — {name}**")
        cols[i].caption(desc)

    st.markdown("---")
    st.markdown("#### Cluster Summary (means)")
    cluster_summary_all = df.groupby('Cluster')[
        ['Age','Recency','Frequency','Monetary','Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend','CLTV_simple']
    ].mean().round(2)
    st.dataframe(cluster_summary_all.style.background_gradient(axis=1))

    st.markdown("#### Cluster counts")
    counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(counts)

    st.markdown("---")
    st.markdown("### Drill-down")
    ccol1, ccol2 = st.columns([2,3])
    with ccol1:
        chosen_cluster = st.selectbox("Cluster to inspect", options=sorted(df['Cluster'].unique()))
        cluster_df = df[df['Cluster'] == chosen_cluster].copy()
        st.write(f"Customers in cluster {chosen_cluster}: {len(cluster_df)}")
        st.download_button(label="Download this cluster", data=cluster_df.to_csv(index=False).encode('utf-8'),
                           file_name=f"cluster_{chosen_cluster}.csv", mime="text/csv")
        # persona
        st.markdown("**Auto Persona Summary**")
        persona_text = generate_persona_text(chosen_cluster, cluster_df)
        st.markdown(persona_text)

    with ccol2:
        sample_ids = cluster_df['CustomerID'].tolist()[:500]
        chosen_customer = st.selectbox("Pick a CustomerID", options=sample_ids)
        cust = df[df['CustomerID'] == chosen_customer].iloc[0]
        st.markdown("**Customer Snapshot**")
        st.write({
            "CustomerID": int(cust['CustomerID']),
            "Age": int(cust['Age']),
            "Gender": cust.get('Gender', 'N/A'),
            "Region": cust['Region'],
            "Cluster": int(cust['Cluster']),
            "Recency (days)": int(cust['Recency']),
            "Frequency (orders/year)": int(cust['Frequency']),
            "Monetary (annual spend)": float(cust['Monetary'])
        })
        st.markdown("**Category Spend Breakdown**")
        spend_df = pd.DataFrame({
            "Category": ["Electronics","Fashion","Grocery","Lifestyle"],
            "Amount": [cust['Electronics_Spend'], cust['Fashion_Spend'], cust['Grocery_Spend'], cust['Lifestyle_Spend']]
        })
        fig_spend = px.pie(spend_df, names='Category', values='Amount', title="Category Spend %", template='plotly_white')
        st.plotly_chart(fig_spend, use_container_width=True)

# -----------------------
# Tab: Prediction
# -----------------------
with tab_prediction:
    st.subheader("Predict new customer's segment + CLTV (quick)")
    st.markdown("Enter details for a new customer to predict cluster and a heuristic CLTV.")

    colA, colB = st.columns(2)
    with colA:
        age_i = st.number_input("Age", min_value=18, max_value=90, value=30)
        recency_i = st.number_input("Recency (days)", min_value=0, max_value=2000, value=60)
    with colB:
        freq_i = st.number_input("Frequency (per year)", min_value=0, max_value=200, value=6)
        monetary_i = st.number_input("Monetary (annual spend ₹)", min_value=0, max_value=1000000, value=5000)

    if st.button("🔍 Predict Segment"):
        new = np.array([[age_i, recency_i, freq_i, monetary_i]])
        new_scaled = scaler.transform(new)

        try:
            pred_cluster = kmeans.predict(new_scaled)[0]
        except Exception:
            kmeans, scaler2, _ = train_kmeans(df, n_clusters=5)
            pred_cluster = kmeans.predict(new_scaled)[0]

        churn_proba_new = min(1.0, recency_i / 365.0)
        avg_order_val_new = monetary_i / (freq_i if freq_i>0 else 1)
        exp_purchases = freq_i
        cltv_new = avg_order_val_new * exp_purchases * (1 - churn_proba_new)

        st.success(f"Predicted Cluster: {pred_cluster}")
        st.write(f"🧠 Estimated Churn Probability: {churn_proba_new:.2f}")
        st.write(f"💰 Estimated CLTV (simple): ₹{cltv_new:,.0f}")

# -----------------------
# Tab: Insights (NEW)
# -----------------------
with tab_insights:
    st.subheader("Automated Insights")
    st.markdown("These cards highlight notable segments and opportunities generated from the data.")

    # compute summary
    agg, sizes = cluster_summary(df)

    avg_cltv_overall = df['CLTV_simple'].mean()
    highest_cltv_cluster = df.groupby('Cluster')['CLTV_simple'].mean().idxmax()
    largest_cluster = sizes.idxmax()
    highest_recency_cluster = df.groupby('Cluster')['Recency'].mean().idxmax()

    # cards row
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Largest Segment", f"Cluster {largest_cluster}", f"{sizes[largest_cluster]} customers")
    r1c2.metric("Highest Avg CLTV", f"Cluster {highest_cltv_cluster}", f"₹{df.groupby('Cluster')['CLTV_simple'].mean().max():,.0f}")
    r1c3.metric("Avg CLTV (overall)", f"₹{avg_cltv_overall:,.0f}")
    r1c4.metric("High Recency (risk)", f"Cluster {highest_recency_cluster}", f"{df.groupby('Cluster')['Recency'].mean().max():.0f} days")

    st.markdown("---")
    st.markdown("### Actionable insights (auto-generated)")
    insights = []
    # insight rules (simple heuristics)
    # 1: high CLTV clusters -> retention
    for cl in sorted(df['Cluster'].unique()):
        cl_df = df[df['Cluster']==cl]
        cl_cltv = cl_df['CLTV_simple'].mean()
        cl_recency = cl_df['Recency'].mean()
        cl_freq = cl_df['Frequency'].mean()
        size = len(cl_df)
        note = f"- **Cluster {cl}**: Avg CLTV ₹{cl_cltv:,.0f}, Avg Recency {cl_recency:.0f} days, Avg Frequency {cl_freq:.1f}. Size: {size}."
        if cl_cltv > avg_cltv_overall:
            note += " Recommendation: *Retention & VIP offerings.*"
        elif cl_recency > 180:
            note += " Recommendation: *Re-engagement campaigns.*"
        else:
            note += " Recommendation: *Upsell / cross-sell opportunities.*"
        insights.append(note)
    for n in insights:
        st.markdown(n)

    st.markdown("---")
    st.markdown("### CLTV distribution by cluster")
    fig_cltv = px.box(df, x='Cluster', y='CLTV_simple', points='outliers', template='plotly_white', title="CLTV Distribution per Cluster")
    st.plotly_chart(fig_cltv, use_container_width=True)

# -----------------------
# Tab: Compare Clusters (NEW)
# -----------------------
with tab_compare:
    st.subheader("Compare Two Clusters Side-by-side")
    cols = st.columns([1,1,2])
    with cols[0]:
        c_left = st.selectbox("Left cluster", options=sorted(df['Cluster'].unique()), index=0, key='left')
    with cols[1]:
        c_right = st.selectbox("Right cluster", options=sorted(df['Cluster'].unique()), index=1 if len(sorted(df['Cluster'].unique()))>1 else 0, key='right')

    if c_left == c_right:
        st.warning("Select two different clusters to compare.")
    else:
        left_df = df[df['Cluster']==c_left]
        right_df = df[df['Cluster']==c_right]

        # summary metrics
        l_metrics = {
            "Count": len(left_df),
            "Avg Age": left_df['Age'].mean(),
            "Avg Monetary": left_df['Monetary'].mean(),
            "Avg CLTV": left_df['CLTV_simple'].mean(),
            "Avg Frequency": left_df['Frequency'].mean(),
            "Avg Recency": left_df['Recency'].mean()
        }
        r_metrics = {
            "Count": len(right_df),
            "Avg Age": right_df['Age'].mean(),
            "Avg Monetary": right_df['Monetary'].mean(),
            "Avg CLTV": right_df['CLTV_simple'].mean(),
            "Avg Frequency": right_df['Frequency'].mean(),
            "Avg Recency": right_df['Recency'].mean()
        }

        # display side-by-side
        lcol, mcol, rcol = st.columns(3)
        with lcol:
            st.markdown(f"**Cluster {c_left}**")
            for k,v in l_metrics.items():
                if "Avg" in k:
                    st.write(f"{k}: {v:.2f}")
                else:
                    st.write(f"{k}: {v}")
        with rcol:
            st.markdown(f"**Cluster {c_right}**")
            for k,v in r_metrics.items():
                if "Avg" in k:
                    st.write(f"{k}: {v:.2f}")
                else:
                    st.write(f"{k}: {v}")

        # Radar / spider plot for category spends (normalized)
        cat_cols = ['Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend']
        left_cat = left_df[cat_cols].mean().values
        right_cat = right_df[cat_cols].mean().values
        # normalize to percent of total
        left_norm = left_cat / (left_cat.sum() + 1e-9) * 100
        right_norm = right_cat / (right_cat.sum() + 1e-9) * 100
        categories = ['Electronics','Fashion','Grocery','Lifestyle']

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=left_norm, theta=categories, fill='toself', name=f'Cluster {c_left}'))
        radar.add_trace(go.Scatterpolar(r=right_norm, theta=categories, fill='toself', name=f'Cluster {c_right}'))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])), showlegend=True, template='plotly_white', title="Category Spend Mix (%)")
        st.plotly_chart(radar, use_container_width=True)

        # bar comparison for core metrics
        comp_df = pd.DataFrame({
            'Metric': ['Avg Monetary','Avg CLTV','Avg Frequency','Avg Recency'],
            f'Cluster {c_left}': [l_metrics['Avg Monetary'], l_metrics['Avg CLTV'], l_metrics['Avg Frequency'], l_metrics['Avg Recency']],
            f'Cluster {c_right}': [r_metrics['Avg Monetary'], r_metrics['Avg CLTV'], r_metrics['Avg Frequency'], r_metrics['Avg Recency']]
        })
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name=f'Cluster {c_left}', x=comp_df['Metric'], y=comp_df[f'Cluster {c_left}']))
        fig_bar.add_trace(go.Bar(name=f'Cluster {c_right}', x=comp_df['Metric'], y=comp_df[f'Cluster {c_right}']))
        fig_bar.update_layout(barmode='group', template='plotly_white', title="Cluster Metric Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption(f"📁 Data: {st.session_state.get('_data_file','unknown')}  •  © 2025 Jatin Bandekar | Clustermap")
