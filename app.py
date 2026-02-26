import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Cybersecurity Threat Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")
    return df

df = load_data()

# ---------------- TITLE ----------------
st.title("🔐 Global Cybersecurity Threats Dashboard (2015-2024)")
st.markdown("Interactive Data Visualization & EDA Platform")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ User Controls")

library = st.sidebar.selectbox(
    "Select Visualization Library",
    ["Seaborn", "Matplotlib", "Pyplot"]
)

chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ["Bar", "Line", "Scatter", "Histogram",
     "Boxplot", "Violin", "KDE", "Heatmap"]
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
y_axis = st.sidebar.selectbox("Select Y-axis", numeric_cols)

# ---------------- CATEGORY FILTER ----------------
filtered_df = df.copy()

if categorical_cols:
    selected_category_column = st.sidebar.selectbox(
        "Select Category Column to Filter (Optional)",
        ["None"] + categorical_cols
    )

    if selected_category_column != "None":
        selected_values = st.sidebar.multiselect(
            "Select Category Values",
            df[selected_category_column].unique(),
            default=df[selected_category_column].unique()
        )

        filtered_df = df[df[selected_category_column].isin(selected_values)]

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Visualization")

fig, ax = plt.subplots(figsize=(9,5))

# -------- SEABORN --------
if library == "Seaborn":

    if chart_type == "Bar":
        sns.barplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)

    elif chart_type == "Line":
        sns.lineplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)

    elif chart_type == "Scatter":
        sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)

    elif chart_type == "Histogram":
        sns.histplot(filtered_df[y_axis], kde=True, ax=ax)

    elif chart_type == "Boxplot":
        sns.boxplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)

    elif chart_type == "Violin":
        sns.violinplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)

    elif chart_type == "KDE":
        sns.kdeplot(filtered_df[y_axis], fill=True, ax=ax)

    elif chart_type == "Heatmap":
        corr = filtered_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

# -------- MATPLOTLIB / PYPLOT --------
elif library in ["Matplotlib", "Pyplot"]:

    if chart_type == "Bar":
        ax.bar(filtered_df[x_axis], filtered_df[y_axis])

    elif chart_type == "Line":
        ax.plot(filtered_df[x_axis], filtered_df[y_axis])

    elif chart_type == "Scatter":
        ax.scatter(filtered_df[x_axis], filtered_df[y_axis])

    elif chart_type == "Histogram":
        ax.hist(filtered_df[y_axis], bins=20)

    elif chart_type == "Boxplot":
        ax.boxplot(filtered_df[y_axis])

    elif chart_type == "Heatmap":
        corr = filtered_df.corr(numeric_only=True)
        im = ax.imshow(corr, cmap="coolwarm")
        plt.colorbar(im)

ax.set_title(f"{chart_type} Chart using {library}")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- DATA SECTION ----------------
st.subheader("🧹 Data Analysis")

with st.expander("View Raw Data (Top 10 Rows)"):
    st.dataframe(df.head(10))

with st.expander("Missing Values Summary"):
    st.write(df.isnull().sum())

with st.expander("Summary Statistics"):
    st.write(df.describe(include='all'))

with st.expander("Correlation Matrix"):
    st.write(df.corr(numeric_only=True))

# ---------------- METRICS ----------------
st.subheader("📈 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(df))
    st.metric("Total Columns", len(df.columns))

with col2:
    st.metric("Numeric Columns", len(numeric_cols))
    st.metric("Categorical Columns", len(categorical_cols))