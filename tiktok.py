import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("TikTok_songs_2022.csv")

colors = ["#09101F","#72DDF7", '#F7AEF8']


def hist(df, x, ax, main_color=colors[1], second_color=colors[0], bins=30):
    sns.histplot(
        data=df,
        x=x,
        bins=bins,
        ax=ax,
        kde=True,
        color=main_color,
        edgecolor=second_color,
        line_kws={"linestyle": '--'},
        linewidth=3
    )

    ax.lines[0].set_color(second_color)
    ax.grid(axis='y', linewidth=0.3, color='black')
    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="x-large")
    ax.set_ylabel("")





cols = ['artist_pop', 'track_pop', 'danceability', 'energy', 'loudness', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

fig, axs = plt.subplots(4, 3, figsize=(30, 25))

for i, col in enumerate(cols):
    row_index = i // 3
    col_index = i % 3
    hist(df, col, axs[row_index][col_index])

fig.suptitle("Histograms of numeric columns of the Dataset", fontsize="xx-large", y=0.92)
plt.show()


def count(df, x, ax, main_color=colors[2], second_color=colors[0]):
    ax.bar(df[x].value_counts().index, 
           df[x].value_counts().values,
           color=main_color, 
           edgecolor=second_color, 
           linewidth=3)
    ax.grid(axis='y', linewidth=0.3, color='black')
    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="x-large")
    ax.set_ylabel("")


cols = ['key', 'mode', 'time_signature']

fig, ax = plt.subplots(1, 3, figsize=(30, 7))

for i, col in enumerate(cols):
    count(df, col, ax[i])
    
fig.suptitle("Count of values for categorical columns", size="xx-large")
plt.show()



df.loc[df["track_pop"] == 0,'track_pop'] = df['track_pop'].mean() 

def scatter(df, x, y, ax, main_color=colors[1], second_color=colors[0]):
    sns.regplot(data=df, 
                x=x, 
                y=y, 
                ax=ax, 
                color=main_color, 
                ci=75,
                scatter_kws={
                    'edgecolor':second_color,
                    'linewidths':1.5,
                    's':50
                },
                line_kws={
                    'color':colors[2],
                    'linewidth':3,
                }
               )
    ax.set_xlabel(x.replace("_", " ").capitalize())
    ax.set_ylabel(y.replace("_", " ").capitalize())
    
    sns.despine(ax=ax)
    ax.grid(axis='x')


fig, ax = plt.subplots(figsize=(10, 6))

scatter(df, 'track_pop', 'artist_pop', ax)

ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

plt.show()
