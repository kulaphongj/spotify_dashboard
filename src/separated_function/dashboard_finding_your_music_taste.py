#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import altair as alt
alt.data_transformers.disable_max_rows()

from vega_datasets import data
import pandas as pd
import numpy as np
import json

import dash_table
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# df_tracks = pd.read_csv('./data/raw/spotify_tracks_genre.csv').iloc[:, 1:]
# print(df_tracks.shape)  # (114000, 20)

# df_tracks_remove_dup = df_tracks.sort_values(['track_name', 'track_id', 'popularity'], ascending=False)
# df_tracks_remove_dup.drop_duplicates(subset=['track_name', 'track_id'], keep='first', inplace=True)
# print(df_tracks_remove_dup.shape)  # (89741, 20)

# save file
# df_tracks_remove_dup.to_csv('./data/preprocessed/df_tracks_remove_dup.csv', index=False, encoding='utf-8-sig')


# In[3]:


# load data
df_tracks = pd.read_csv('./data/preprocessed/df_tracks_remove_dup.csv')


# In[4]:


# df_tracks.head(3)


# In[5]:


# # prepare data
# list_artists = df_tracks['artists'].unique().tolist()
# list_artists = [str(artist) for artist in list_artists]
# list_artists.sort()
# list_track_name = df_tracks['track_name'].unique().tolist()
# list_track_name = [str(song) for song in list_track_name]
# list_track_name.sort()
# list_track_genre = df_tracks['track_genre'].unique().tolist()
# list_track_genre = [str(genre) for genre in list_track_genre]
# list_track_genre.sort()

# list_find_range_cols = ['popularity', 'danceability', 'energy', 'loudness',
#                          'speechiness', 'acousticness', 
#                        'instrumentalness', 'liveness', 'valence', 'tempo',]

# # fucntion for getting 100 step range of data to build list of values of slicer
# def build_list_values(df, column):
#     vmin = df[column].min()
#     vmax = df[column].max()
#     steps = (vmax - vmin)/100
    
#     if df[column].dtypes==int:
#         steps = int(steps)
#         vmax = vmax+steps
#         list_out = list(range(vmin, vmax, steps))
#     elif df[column].dtypes==float:
#         vmax = vmax+steps
#         list_out = list(np.arange(vmin, vmax, steps))
#         list_out = [round(i, 2) for i in list_out]
#     list_out.sort()
#     return list_out

# dict_cols_val = {}
# for col in list_find_range_cols:
#     dict_cols_val[col] = build_list_values(df_tracks, col)

# # Convert and write JSON object to file
# with open("./data/preprocessed/dict_cols_val.json", "w") as outfile: 
#     json.dump(dict_cols_val, outfile)

# dict_artist = {'artists':list_artists}
# dict_trackname = {'track_name':list_track_name}
# dict_genre = {'track_genre':list_track_genre}

# # Convert and write JSON object to file
# with open("./data/preprocessed/artist.json", "w") as outfile: 
#     json.dump(dict_artist, outfile)
    
#     # Convert and write JSON object to file
# with open("./data/preprocessed/track_genre.json", "w") as outfile: 
#     json.dump(dict_genre, outfile)
    
#     # Convert and write JSON object to file
# with open("./data/preprocessed/track_name.json", "w") as outfile: 
#     json.dump(dict_trackname, outfile)


# In[130]:


# load data
f = open('./data/preprocessed/track_genre.json',) 
list_track_genre = json.load(f) ['track_genre']
f.close() 
f = open('./data/preprocessed/track_name.json',) 
list_track_name = json.load(f) ['track_name']
f.close()  
f = open('./data/preprocessed/artist.json',) 
list_artists = json.load(f) ['artists']
f.close() 

# print(len(list_artists), len(list_track_name), len(list_track_genre))

# value for slicer
f = open('./data/preprocessed/dict_cols_val.json',) 
dict_cols_val = json.load(f)
f.close() 

# print(len(dict_cols_val))


# In[131]:


# # comment for production
# list_artists = list_artists[:20]
# list_track_name = list_track_name[:5]


# In[132]:


# Specify the columns for the radar chart
list_cols_radar = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
           'instrumentalness', 'liveness', 'valence', 'tempo']
# default genre for showing radar chart
selected_genres_default = ['tango', 'acoustic', 'black-metal', 'j-idol', 'anime', 
                     'idm', 'comedy', 'pop', 'blues', 'disco', 'k-pop', 'romance', 'death-metal']


# list for stats table
list_stats_dsp = ['popularity', 'danceability', 'energy', 
                 'loudness', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'tempo']
df_table = df_tracks[list_stats_dsp].describe().T[['min', 'mean', 'max']].reset_index()
df_table.columns = ['Statistics', 'Min', 'Mean', 'Max']
df_table = df_table.sort_values('Statistics')
df_table = df_table.round(2)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
       
    html.Br(),
    dbc.Row([
        # Col1 : filter and Logo
            dbc.Col([
                dbc.Card([
                    # Row 1: Filter
                    dbc.Row([
                        # Col1: filter genre
                        dbc.Col([
                                html.P("Genre", style={'margin-left': '5px', 'margin-top': '10px'}),
                                dcc.Dropdown(
                                            id="genre-filter",
                                            options=[{'label': genre, 'value': genre} for genre in list_track_genre],
                                            value=[],
                                            multi=True,
                                            placeholder="Select an Genre",
                                            style={'width': '3',  
                                                   'min-height': '67vh',
                                                   'margin-left': '3px',
                                                   'margin-bottom': '3px'} 
                                ) 
                        ], width=6),

                        # Col2: filter artist and track name
                        dbc.Col([
                            # Row1: filter artist
                            dbc.Row([
                                    html.P("Artist", style={'margin-left': '5px', 'margin-top': '10px'}),
                                    dcc.Dropdown(
                                                id="artist-filter",
                                                options=[{'label': artist, 'value': artist} for artist in list_artists],
                                                value=[],
                                                multi=True,
                                                placeholder="Select an artist",
                                                style={'width': '3',  
                                                       'min-height': '30vh'} 
                                    ) 
                            ]),            
                            # Row2: filter track name
                            dbc.Row([
                                    html.P(),
                                    html.P("Track name", style={'margin-left': '5px', 'margin-top': '10px'}),
                                    dcc.Dropdown(
                                                id="trackname-filter",
                                                options=[{'label': song, 'value': song} for song in list_track_name],
                                                value=[],
                                                multi=True,
                                                placeholder="Select an Track Name",
                                                style={'width': '3',  
                                                       'min-height': '30vh',
                                                       'margin-bottom': '3px'
                                                      }   
                                    )
                            ])
                        ], width=6, style={'margin-left': '-7px'})
                    ]),
                ], color='light'),
                    
                # Row 2 : Logo
                dbc.Row([
                    html.Br(),
                    html.H1('Spotify', style={'fontSize': 80, 'textAlign': 'center', 'marginTop': '25px',
                                               'color': 'green', 'height': '100px'}),
                    html.H5('Find Your Music Taste', style={'fontSize': 20, 'textAlign': 'center',
                                                            'color': 'green'})
                ])

            ], width=4),

        
        # Col2 : Charts
        dbc.Col([
            # Row1: Table and Pie Charts
            dbc.Row([
                # Col1: Table of song
                dbc.Col([
                    html.P("Your Music Taste", style={'margin-left': '5px', 'margin-top': '10px'}),
                    dbc.Card([
                        dash_table.DataTable(
                                         id='stats-table',
                                         columns=[{'name': col, 'id': col} for col in df_table.columns],
                                         data=df_table.to_dict('records'),
                                         style_table={'width': '4', 'height': '320px', 'marginTop': '15px'},
                                         style_cell={'font_size': '14px', 'whiteSpace': 'normal','word-wrap': 'break-word',
                                                    'textAlign': 'center', 'minWidth': '60px', 'maxWidth': '60px', 
                                                    'backgroundColor': 'transparent'}  ,
                                         style_data={'border': '0px'},
                                         style_header={'border': '0px', 'fontWeight': 'bold', 'font-size': '18px'},
                                         style_data_conditional=[
                                                    {'if': {'column_id': 'Statistics'}, 
                                                     'textAlign': 'center', 'minWidth': '140px', 'maxWidth': '140px' }]                  
                        )
                    ], color='light')
                ], width=6),
                
                # Col2: Pie Chart
                dbc.Col([
                    html.P("Genre Proportion", style={'margin-left': '5px', 'margin-top': '10px'}),
                    dbc.Card([
                        html.Iframe(
                                    id='pie-chart',
                                    style={'border-width': '0', 'width': '100%', 'height': '335px'}
                        )
                    ], color="light")
                ], width=6)
            ]),
            
            # Row2: Radar Charts
            dbc.Row([
               #Col1: Radar Chart
                dbc.Col([
                    html.P("Music Taste Status", style={'margin-left': '5px', 'margin-top': '10px'}),
                    dbc.Card([
                        dcc.Graph(id='radar-chart')       
                    ], color="light")
                ], width=12)
            ]),        
        ], width=8)
        
        


#                 #Col2: Bar Chart
#                 dbc.Col([
#                     html.P("Tempo Distribution"),
#                     dbc.Card([
#                         html.Iframe(
#                                     id='bar-chart',
#                                     style={'border-width': '0', 'width': '100%', 'height': '320px'}
#                         )
#                     ], color="light")
#                 ], width=6)
    ])    
])
    



@app.callback(
    Output('stats-table', 'data'),  # Statistics table
    Output('pie-chart', 'srcDoc'),  # Genre Pie Chart
#     Output('bar-chart', 'srcDoc'),  # Tempo Bar Chart
    Input('genre-filter', 'value'),
    Input('trackname-filter', 'value'),
    Input('artist-filter', 'value'))
def filter_genre(slct_genre, slct_track, slct_artist):
    # filter data
    df_filt = df_tracks.copy()
    if len(slct_genre)>0:
        df_filt = df_tracks[df_tracks['track_genre'].isin(slct_genre)]
    if len(slct_track)>0:
        df_filt = df_tracks[df_tracks['track_name'].isin(slct_track)]
    if len(slct_artist)>0:
        df_filt = df_tracks[df_tracks['artists'].isin(slct_artist)]
    

    # stats table
    df_table = df_filt[list_stats_dsp].describe().T[['min', 'mean', 'max']].reset_index()
    df_table.columns = ['Statistics', 'Min', 'Mean', 'Max']
    df_table = df_table.sort_values('Statistics')
    df_table = df_table.round(2)
    
    # genre pie chart
    df_pie = pd.concat([df_filt['track_genre'].value_counts(normalize=True),
                   df_filt['track_genre'].value_counts()], axis=1)
    df_pie = df_pie.reset_index()
    df_pie.columns = ['Genre', 'Percentage', 'Count']

    # limit number of Genre to 10
    if len(df_pie)>10:
        df_pie.loc[9:, "Genre"] = 'Others'
        df_pie = df_pie.groupby(["Genre"]).sum().reset_index()
#     df_pie = df_pie.round(2)
       
    chart_pie = alt.Chart(df_pie).mark_arc(innerRadius=0).encode(
                        theta=alt.Theta(field="Percentage", type="quantitative"),
                        color=alt.Color(field="Genre", type="nominal"),
                        tooltip=[alt.Tooltip("Genre:N"), 
                                 alt.Tooltip("Percentage:Q", format='.2%'), 
                                 alt.Tooltip("Count:Q", format=',')]
                )

    text = chart_pie.mark_text(radius=135, size=12, align="center").encode(
                text=alt.Text("Percentage:Q",format=".1%",),
            )
    
    
    chart_pie_t = (chart_pie + text).properties(width=250, height=300, background='transparent').configure_view(strokeWidth=0)
    
#     # tempo bar chart
#     df_bar = df_filt[['tempo']]
#     chart_bar = alt.Chart(df_bar).mark_bar().encode(
#                         alt.X("tempo:Q", bin=alt.Bin(maxbins=20)),
#                         y='count()',
#                 ).properties(width=250, height=250, background='transparent')
    
    return df_table.to_dict('records'), chart_pie_t.to_html()#, chart_bar.to_html()





# Radar chart with plotly
@app.callback(
     dash.dependencies.Output('radar-chart', 'figure'),
     dash.dependencies.Input('genre-filter', 'value'),
     dash.dependencies.Input('trackname-filter', 'value'),
     dash.dependencies.Input('artist-filter', 'value')
)
def update_radar_chart(slct_genre, slct_track, slct_artist):
    # filter data
    df_filt = df_tracks.copy()
    if len(slct_genre)>0:
        df_filt = df_tracks[df_tracks['track_genre'].isin(slct_genre)]
    if len(slct_track)>0:
        df_filt = df_tracks[df_tracks['track_name'].isin(slct_track)]
    if len(slct_artist)>0:
        df_filt = df_tracks[df_tracks['artists'].isin(slct_artist)]
        
    # Group by genre and calculate the mean of each metric
    mean_metrics_by_genre = df_filt.groupby('track_genre')[list_cols_radar].mean()

    # Filter the mean metrics data for the selected genres
    selected_genres = df_filt['track_genre'].unique().tolist()
    mean_metrics_selected_genres = mean_metrics_by_genre.loc[selected_genres]

    # Min-max scaling for 'loudness' and 'tempo' columns
    scaler = MinMaxScaler()
    mean_metrics_selected_genres_scaled = mean_metrics_selected_genres.copy()
    mean_metrics_selected_genres_scaled[['loudness', 'tempo']] = scaler.fit_transform(mean_metrics_selected_genres[['loudness', 'tempo']])

    # Transpose the scaled data for easier plotting
    mean_metrics_selected_genres_transposed_scaled = mean_metrics_selected_genres_scaled.transpose()

    # Create traces for each genre
    data = []
    for genre in selected_genres:
        data.append(go.Scatterpolar(
            r=mean_metrics_selected_genres_transposed_scaled[genre].tolist(),
            theta=list_cols_radar,
            fill='toself',
            name=genre
        ))

    # Create the layout
    layout = go.Layout(
#         title="Average Music Metrics by Genre (Radar Chart)",
        polar=dict(
            radialaxis=dict(visible=True),
            
            
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=10, r=10, t=30, b=30)
    )

    # Create the radar chart figure
    fig = go.Figure(data=data, layout=layout)

    return fig


# In[133]:


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




