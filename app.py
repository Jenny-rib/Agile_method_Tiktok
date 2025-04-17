import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

# Model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.set_page_config(page_title="Track Popularity Predictor", layout="wide")
st.title("🎵 Track Popularity Predictor")

# CSV
df = pd.read_csv("TikTok_songs_2022.csv")

# Setting tab
tab1, tab2 = st.tabs(["🎯 Prediction", "📊 Visualization"])

# Tab 1
with tab1:
    st.header("💻 Enter the audio characteristics of the song")
    feature_names = [
        "Artist Popularity", "Danceability", "Energy", "Loudness", "Mode", "Key", "Speechiness", "Acousticness",
        "Liveness", "Valence", "Tempo", "Time Signature", "Duration (ms)"
    ]

    features = []
    for name in feature_names:
        value = st.number_input(name, format="%.5f")
        features.append(value)

    if st.button("Predict song popularity"):
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"The predicted Track Popularity is：{round(prediction, 2)}")

        # 导出结果按钮
        csv_data = pd.DataFrame([features + [round(prediction, 2)]], columns=feature_names + ["Predicted Track Popularity"])
        csv_buffer = BytesIO()
        csv_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export the prediction result as CSV",
            data=csv_buffer.getvalue(),
            file_name="prediction_result.csv",
            mime="text/csv"
        )

# Tab 2
with tab2:

    # 可视化下拉选择器
    chart_option = st.selectbox(
        "Select the chart to display：",
        [
            "Feature correlation heat map",
            "Top 10 Most Popular Artists",
            "Top 10 TikTok Popular Artists (Number of Songs)",
            "Top 10 Most Popular Songs"
        ]
    )

    if chart_option == "Feature correlation heat map":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        selected_cols = st.multiselect("Select the characteristics to display relevance：", options=numeric_cols, default=numeric_cols[:6])
        if selected_cols:
            corr_df = df[selected_cols].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
            sns.heatmap(
                corr_df, annot=True, cmap="coolwarm",
                linewidths=0.5, ax=ax_corr,
                annot_kws={"fontsize": 8},
                cbar_kws={"shrink": 0.5}
            )
            ax_corr.set_title("Correlation Matrix", fontsize=12, pad=10)
            ax_corr.tick_params(axis='both', labelsize=8)
            fig_corr.tight_layout(pad=1.0)
            st.pyplot(fig_corr)

    elif chart_option == "Top 10 Most Popular Artists":
        artists_pop_sorted = df.loc[:, ['artist_name', 'artist_pop']].drop_duplicates('artist_name')
        top10_artists = artists_pop_sorted.sort_values('artist_pop', ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=top10_artists, x='artist_pop', y='artist_name', palette='Blues_r', ax=ax1)
        ax1.set_title("Top 10 Artists by Popularity", fontsize=12)
        ax1.tick_params(labelsize=8)
        fig1.tight_layout(pad=1.0)
        st.pyplot(fig1)

    elif chart_option == "Top 10 TikTok Popular Artists (Number of Songs)":
        artists_top_hits = df['artist_name'].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(x=artists_top_hits.values, y=artists_top_hits.index, palette='Purples_r', ax=ax2)
        ax2.set_title("Top 10 Artists by Number of Tracks on TikTok", fontsize=12)
        ax2.tick_params(labelsize=8)
        fig2.tight_layout(pad=1.0)
        st.pyplot(fig2)

    elif chart_option == "Top 10 Most Popular Songs":
        tracks_pop_sorted = df.loc[:, ['track_name', 'track_pop']].sort_values('track_pop', ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=tracks_pop_sorted, x='track_pop', y='track_name', palette='Oranges_r', ax=ax3)
        ax3.set_title("Top 10 Tracks by Popularity", fontsize=12)
        ax3.tick_params(labelsize=8)
        fig3.tight_layout(pad=1.0)
        st.pyplot(fig3)