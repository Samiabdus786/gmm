import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

st.set_page_config(
    page_title="Mall Customer GMM Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI Customer Segmentation & Marketing Recommendation System")
st.write("Gaussian Mixture Model (GMM) Based Customer Segmentation Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

st.sidebar.success("Dataset Loaded Successfully ✔")

feature_columns = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("⚙️ GMM Parameters")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

clusters = gmm.predict(X_scaled)

df["Cluster"] = clusters

st.sidebar.header("🧍 Enter New Customer Details")

age = st.sidebar.slider(
    "Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    30
)

income = st.sidebar.slider(
    "Annual Income",
    int(df["Annual Income (k$)"].min()),
    int(df["Annual Income (k$)"].max()),
    60
)

spending = st.sidebar.slider(
    "Spending Score",
    1,
    100,
    50
)

input_data = pd.DataFrame(
    [[age, income, spending]],
    columns=feature_columns
)

scaled_input = scaler.transform(input_data)

predicted_cluster = gmm.predict(scaled_input)[0]

cluster_probabilities = gmm.predict_proba(scaled_input)[0]

st.subheader("🎯 Cluster Prediction")

st.success(
    f"This customer belongs to Cluster {predicted_cluster}"
)

st.subheader("📊 Cluster Membership Probabilities")

prob_df = pd.DataFrame({
    "Cluster": [f"Cluster {i}" for i in range(n_clusters)],
    "Probability": cluster_probabilities
})

st.dataframe(prob_df, use_container_width=True)

def describe_customer(income, spending):

    if income > 70 and spending > 70:
        return (
            "💎 Premium Customer",
            "Offer VIP memberships, premium services and exclusive deals."
        )

    elif income > 70 and spending < 40:
        return (
            "🧠 Careful Rich Customer",
            "Target with personalized luxury offers and email campaigns."
        )

    elif income < 40 and spending > 70:
        return (
            "🛒 Impulsive Buyer",
            "Provide flash sales, combo offers and discounts."
        )

    elif income < 40 and spending < 40:
        return (
            "💰 Budget Customer",
            "Offer coupons, cashback and seasonal promotions."
        )

    else:
        return (
            "🙂 Average Customer",
            "Provide loyalty rewards and regular advertisements."
        )

customer_type, strategy = describe_customer(
    income,
    spending
)

st.info(f"Customer Type: {customer_type}")
st.warning(f"Recommended Marketing Strategy: {strategy}")

st.subheader("👤 Customer Profile")

c1, c2, c3 = st.columns(3)

c1.metric("Age", age)
c2.metric("Income", income)
c3.metric("Spending Score", spending)

st.subheader("📊 Cluster Distribution")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

fig_pie = go.Figure(
    data=[
        go.Pie(
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            values=cluster_counts.values,
            textinfo="label+percent"
        )
    ]
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

st.subheader("📈 Customer Segments Visualization")

fig_scatter = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    hover_data=["Age"],
    title="GMM Customer Segments"
)

fig_scatter.add_scatter(
    x=[income],
    y=[spending],
    mode="markers",
    marker=dict(
        color="red",
        size=15,
        symbol="diamond"
    ),
    name="New Customer"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

st.subheader("📊 Cluster Behavior Comparison")

cluster_summary = (
    df.groupby("Cluster")
    .mean(numeric_only=True)
)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

fig_bar = px.bar(
    cluster_summary,
    x=cluster_summary.index,
    y=[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ],
    barmode="group",
    title="Average Income & Spending by Cluster"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

st.subheader("📉 Model Quality Metrics")

bic = gmm.bic(X_scaled)
aic = gmm.aic(X_scaled)

m1, m2 = st.columns(2)

m1.metric("BIC Score", round(bic, 2))
m2.metric("AIC Score", round(aic, 2))

st.subheader("📌 Dataset Statistics")

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Total Customers",
    len(df)
)

d2.metric(
    "Clusters",
    n_clusters
)

d3.metric(
    "Average Age",
    round(df["Age"].mean(), 1)
)

d4.metric(
    "Average Income",
    round(df["Annual Income (k$)"].mean(), 1)
)

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.markdown("---")
st.caption(
    "Powered by Gaussian Mixture Model (GMM) Clustering"
)