#!/usr/bin/env python
# coding: utf-8

import dash
from dash import dcc, html, Input, Output, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import altair as alt
alt.data_transformers.disable_max_rows()

from vega_datasets import data
import pandas as pd
import numpy as np
import json

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os
import pycountry

import requests
from bs4 import BeautifulSoup
import urllib.request
import time


# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Music Explorer with Spotify")
server = app.server


# # Tabs

# ## Tab 1

# Data Preprocessing Tab1
# load data
df_tracks = pd.read_csv('./data/preprocessed/df_tracks_interestgenre.csv')

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

# # comment for production 
# # uncomment for making the faster dashboard
list_artists = list_artists[:20]
list_track_name = list_track_name[:5]

def filter_taste(slct_genre, slct_track, slct_artist):
    df_filt = df_tracks.copy()
    # filter data
    ## filter all
    if (len(slct_genre)>0)&(len(slct_track)>0)&(len(slct_artist)>0):
        cond_genre = df_tracks['track_genre'].isin(slct_genre)
        cond_track = df_tracks['track_name'].isin(slct_track)
        cond_artist = df_tracks['artists'].isin(slct_artist)
        df_filt = df_tracks[cond_genre|cond_track|cond_artist]
    ## filter only genre
    elif (len(slct_genre)>0)&(len(slct_track)==0)&(len(slct_artist)==0):
        df_filt = df_tracks[df_tracks['track_genre'].isin(slct_genre)]
    ## filter genre and trackname
    elif (len(slct_genre)>0)&(len(slct_track)>0)&(len(slct_artist)==0):
        cond_genre = df_tracks['track_genre'].isin(slct_genre)
        cond_track = df_tracks['track_name'].isin(slct_track)
        df_filt = df_tracks[cond_genre|cond_track]
    ## filter only trackname
    elif (len(slct_genre)==0)&(len(slct_track)>0)&(len(slct_artist)==0):
        df_filt = df_tracks[df_tracks['track_name'].isin(slct_track)]
    ## filter trackname and artist
    elif (len(slct_genre)==0)&(len(slct_track)>0)&(len(slct_artist)>0):
        cond_track = df_tracks['track_name'].isin(slct_track)
        cond_artist = df_tracks['artists'].isin(slct_artist)
        df_filt = df_tracks[cond_track|cond_artist] 
    ## filter only artist
    elif (len(slct_genre)==0)&(len(slct_genre)==0)&(len(slct_artist)>0):
        df_filt = df_tracks[df_tracks['artists'].isin(slct_artist)]
    ## filter genre and artist
    elif (len(slct_genre)>0)&(len(slct_track)==0)&(len(slct_artist)>0):
        cond_genre = df_tracks['track_genre'].isin(slct_genre)
        cond_artist = df_tracks['artists'].isin(slct_artist)
        df_filt = df_tracks[cond_genre|cond_artist]
    
    return df_filt

# Specify the columns for the radar chart
list_cols_radar = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
           'instrumentalness', 'liveness', 'valence', 'tempo']
# default genre for showing radar chart
selected_genres_default = ['pop', 'hip-hop', 'rock-n-roll',  
                             'rock', 'edm', 'r-n-b', 'country',
                             'latin', 'indie', 'k-pop', 'metal',
                             'classical', 'jazz', 'blues', 'folk', 'reggae', 'soul']


# list for stats table
list_stats_dsp = ['popularity', 'danceability', 'energy', 
                 'loudness', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'tempo']
df_table = df_tracks[list_stats_dsp].describe().T[['min', 'mean', 'max']].reset_index()
df_table.columns = ['Statistics', 'Min', 'Mean', 'Max']
df_table = df_table.sort_values('Statistics')
df_table = df_table.round(2)


# app.layout = dbc.Container([
tab1_content = html.Div([
    html.Br(),
    dbc.Row([
        # Col1 : filter and Logo
            dbc.Col([
                # Row 1: Filter
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Filter", 
                                           style={'backgroundColor': '#68A58C',
                                                  'fontWeight': 'bold', 'color': 'white',
                                                  'font-size': '18px'}),
                            dbc.CardBody([
                                # Row 1: Filter
                                dbc.Row([
                                    dbc.Col([
                                        # filter track name and artist
                                        dbc.Row([
                                            # Col1: filter trackname
                                            dbc.Col([
                                                html.P("Track name", style={'margin-left': '5px', 'margin-top': '10px'}),
                                                dcc.Dropdown(
                                                            id="trackname-filter",
                                                            options=[{'label': song, 'value': song} for song in list_track_name],
                                                            value=[],
                                                            multi=True,
                                                            optionHeight=110,
                                                            placeholder="Select an Track Name",
                                                            style={'width': '6',  
                                                                   'min-height': '28vh'}   
                                                )
                                            ], width=6),

                                            # Col2: filter artist
                                            dbc.Col([
                                                html.P("Artist", style={'margin-left': '5px', 'margin-top': '10px'}),
                                                dcc.Dropdown(
                                                            id="artist-filter",
                                                            options=[{'label': artist, 'value': artist} for artist in list_artists],
                                                            value=[],
                                                            multi=True,
                                                            optionHeight=110,
                                                            placeholder="Select an Artist",
                                                            style={'width': '6',  
                                                                   'min-height': '28vh'} 
                                                ) 
                                            ], width=6),


                                        ]),

                                        # filter genre
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
                                                                style={'width': '12',  
                                                                       'min-height': '20vh',
                                                                       'margin-left': '3px',
                                                                       'margin-bottom': '3px'} 
                                                    ),
                                                html.Div(id='hidden-data', style={'display': 'none'})
                                            ], width=12)      

                                        ]),
                                    ], width=12)
                                ]),
                            ])
                        ], color='light'),
                    ], width=12)
                ]),
                    
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
                    dbc.Card([
                        dbc.CardHeader("Your Music Taste", 
                                           style={'backgroundColor': '#68A58C',
                                                  'fontWeight': 'bold', 'color': 'white',
                                                  'font-size': '18px'}),          
                        dbc.CardBody([
                            dash_table.DataTable(
                                             id='stats-table',
                                             columns=[{'name': col, 'id': col} for col in df_table.columns],
                                             data=df_table.to_dict('records'),
                                             style_table={'width': '6', 'height': '330px', 
#                                                           'marginTop': '15px',
                                                          'overflowX': 'auto'},
                                             style_cell={'font_size': '14px', 'whiteSpace': 'normal',
                                                         'word-wrap': 'break-word',
                                                        'textAlign': 'center', 'minWidth': '60px', 
                                                         'maxWidth': '60px', 
                                                        'backgroundColor': 'transparent'}  ,
                                             style_data={'border': '0px'},
                                             style_header={'border': '0px', 'fontWeight': 'bold', 
                                                           'font-size': '18px'},
                                             style_data_conditional=[
                                                        {'if': {'column_id': 'Statistics'}, 
                                                         'textAlign': 'center', 'minWidth': '140px', 
                                                         'maxWidth': '140px' }]                  
                            )
                        ], style={'height': '330px'})
                    ], color='light')
                ], width=6),
                
                # Col2: Pie Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Genre Proportion", 
                                           style={'backgroundColor': '#68A58C',
                                                  'fontWeight': 'bold', 'color': 'white',
                                                  'font-size': '18px'}),
                        dbc.CardBody([
#                             html.Iframe(
#                                         id='pie-chart',
#                                         style={'border-width': '0', 'width': '100%', 'height': '330px'}
#                             )
                            dcc.Graph(
                                id='pie-chart',
#                                 config={'displayModeBar': False}, # Hide the mode bar
                                style={'height': '100%', 'border-width': '0'}
                            ),
                        ], style={'height': '330px'})
                    ], color="light")
                ], width=6)
            ], className="gx-3"),
            
            # Row2: Radar Charts
            dbc.Row([
               #Col1: Radar Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Music Taste Status", 
                                           style={'backgroundColor': '#68A58C',
                                                  'fontWeight': 'bold', 'color': 'white',
                                                  'font-size': '18px'}),
                        dbc.CardBody([
                            dcc.Graph(id='radar-chart')
                        ])
                    ], color="light", style={'margin-top': '16px'})
                ], width=12)
            ]),        
        ], width=8)
    ], className="gx-3")   
])

@app.callback(
    Output('stats-table', 'data'),  # Statistics table
#     Output('pie-chart', 'srcDoc'),  # Genre Pie Chart
    Output('pie-chart', 'figure'),  # Genre Pie Chart
    Input('genre-filter', 'value'),
    Input('trackname-filter', 'value'),
    Input('artist-filter', 'value'))
def filter_genre(slct_genre, slct_track, slct_artist):
    
    df_filt = filter_taste(slct_genre, slct_track, slct_artist)
    
    # stats table
    df_table = df_filt[list_stats_dsp].describe().T[['min', 'mean', 'max']].reset_index()
    df_table.columns = ['Statistics', 'Min', 'Mean', 'Max']
    df_table['Statistics'] = df_table['Statistics'].str.capitalize()
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
       
#     chart_pie = alt.Chart(df_pie).mark_arc(innerRadius=0).encode(
#                         theta=alt.Theta(field="Percentage", type="quantitative"),
#                         color=alt.Color(field="Genre", type="nominal"),
#                         tooltip=[alt.Tooltip("Genre:N"), 
#                                  alt.Tooltip("Percentage:Q", format='.2%'), 
#                                  alt.Tooltip("Count:Q", format=',')]
#                 )

#     text = chart_pie.mark_text(radius=135, size=12, align="center").encode(
#                 text=alt.Text("Percentage:Q",format=".1%",),
#             )
    
    
#     chart_pie_t = (chart_pie + text).properties(width=230, height=270, 
#                                                 background='transparent').configure_view(strokeWidth=0)
    
#     return df_table.to_dict('records'), chart_pie_t.to_html()

    
#     Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=df_pie['Genre'],
        values=df_pie['Count'],
    )])
    fig.update_traces(textposition='inside', hoverinfo="label+value")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', 
                      margin=dict(t=0, b=0, l=0, r=0),
                      paper_bgcolor='rgba(0,0,0,0)')
        
    return df_table.to_dict('records'), fig




# Radar chart with plotly
@app.callback(
     dash.dependencies.Output('radar-chart', 'figure'),
     dash.dependencies.Input('genre-filter', 'value'),
     dash.dependencies.Input('trackname-filter', 'value'),
     dash.dependencies.Input('artist-filter', 'value')
)
def update_radar_chart(slct_genre, slct_track, slct_artist):
    
    df_filt = filter_taste(slct_genre, slct_track, slct_artist)
        
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


# ## Tab 2

df = df_tracks.copy()
df['artists'] = df['artists'].str.split(';').str[0]
# Helper functions
def generate_marks(feature_min, feature_max):
    step = max((feature_max - feature_min) / 5, 1) 
    if feature_max<=1:
        return {feature_min: f"{feature_min:.2f}", feature_max: f"{feature_max:.2f}"}
    else:
        return {i: f"{i:.2f}" for i in range(int(feature_min), int(feature_max) + 1, int(step))}

def normalize(df, features):
    result = df.copy()
    for feature_name in features:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Tooltips for each slider
feature_tooltips = {
    'danceability': 'How suitable a track is for dancing based on a combination of musical elements.',
    'energy': 'A measure of intensity and activity.',
    'speechiness': 'The presence of spoken words in a track.',
    'acousticness': 'A measure of how acoustic a track is.',
    'instrumentalness': 'The likelihood that a track contains no vocal content.',
    'liveness': 'The presence of an audience in the recording.',
    'valence': 'The musical positiveness conveyed by a track.',
    'tempo': 'The overall estimated tempo of a track in beats per minute (BPM).'
}

# Layout adjustments
tab2_content = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Discover New Music", className="text-center mb-4"), width=12),
    ]),
    dbc.Row([
        # Left column
        dbc.Col([
            # Genre Selection Card
            dbc.Card([
                dbc.CardHeader("Step1: Select Genre", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='genre-dropdown',
                        options=[{'label': genre, 'value': genre} for genre in df['track_genre'].unique()],
                        #value=[df['track_genre'].unique()[1]],
                        value=['pop','k-pop'],
                        multi=True
                    ),
                ])
            ], style={'marginBottom': '20px'}),

            # Music Features Card
            dbc.Card([
                dbc.CardHeader("Step2: Change Music Feature Ranges", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label(f"{feature.capitalize()}:", id=f"label-{feature}"),
                            ]),
                            dcc.RangeSlider(
                                id=f'{feature}-slider',
                                min=df[feature].min(),
                                max=df[feature].max(),
                                step=0.01,
                                marks=generate_marks(df[feature].min(), df[feature].max()),
                                value=[df[feature].min(), df[feature].max()],
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            dbc.Tooltip(
                                feature_tooltips[feature],
                                target=f"label-{feature}",
                            ),
                        ], style={'padding': '10px', 'margin': '10px 0'})
                        for feature in [
                            'danceability', 'energy', 'speechiness',
                            'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo'
                        ]
                    ])
                ]),
            ], style={'marginBottom': '20px'}),
        ], md=4),

        # Right column for Graphs and Tables
        dbc.Col([
            # Selected Features and Trends Card
            dbc.Card([
                dbc.CardHeader("Step3: Observe Selected Music Feature Distribution", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                dbc.CardBody([
                    dcc.Graph(id='parallel-coordinates-plot'),
                ])
            ], style={'marginBottom': '20px'}),
            #Songs tables
            dbc.Card([
                dbc.CardHeader("Step4: Popular Songs Based On Your Selections", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='songs-table',
                        columns=[
                            {"name": "Track Name", "id": "track_name"},
                            {"name": "Popularity", "id": "popularity"},
                            {"name": "Artists", "id": "artists"},
                            {"name": "Spotify Link", "id": "track_id", "presentation": "markdown"},
                        ],
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        page_size=10,
                        style_cell={'textAlign': 'center'},
                        style_header={
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                    )
                ])
            ], style={'marginBottom': '20px'}),
        ], md=8),
    ]),
])


@app.callback(
    [Output('parallel-coordinates-plot', 'figure'),
     Output('songs-table', 'data')],
    [Input(f'{feature}-slider', 'value') for feature in [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]] + [Input('genre-dropdown', 'value')]
)
def update_content(*args):
    # Split inputs into slider values and selected genres
    slider_values = args[:-1]  # Extract slider values for features
    selected_genres = args[-1]  # Extract selected genre(s) from dropdown

    # Filter the DataFrame based on selected genres
    filtered_df = df[df['track_genre'].isin(selected_genres)]

    # Further filter the DataFrame based on slider values for each feature
    features = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    for i, feature in enumerate(features):
        min_val, max_val = slider_values[i]
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

    # Normalize the filtered data for visual comparison
    normalized_df = normalize(filtered_df, features)

    # Create a Parallel Coordinates Plot for visualizing the songs
    parallel_coordinates_figure = go.Figure(data=go.Parcoords(
        line=dict(color=normalized_df['popularity'],
                  colorscale=[[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']],
                  showscale=True),
        dimensions=[{'label': col, 'values': normalized_df[col]} for col in features]
    ))
    parallel_coordinates_figure.update_layout(title_text='Parallel Coordinates Plot for Selected Features')

    # Prepare data for the songs table
    songs_table_data = filtered_df[['track_name', 'popularity', 'artists', 'track_id']]\
        .sort_values(by='popularity', ascending=False)
    songs_table_data['track_id'] = songs_table_data['track_id']\
        .apply(lambda x: f"[Listen on Spotify](https://open.spotify.com/track/{x})")
    songs_table_data = songs_table_data.to_dict('records')

    return parallel_coordinates_figure, songs_table_data

# ## Tab 3

# Data Preprocessing Tab3
# Load the data
spotify_data_countries = pd.read_csv('./data/raw/spotify_tracks_country.csv')
# spotify_data_genres = pd.read_csv('./data/raw/spotify_tracks_genre.csv')

# Define a mapping of your country codes to ISO Alpha-3 codes
country_mapping = {
    'AE': 'ARE', 'AR': 'ARG', 'AT': 'AUT', 'AU': 'AUS', 'BE': 'BEL',
    'BG': 'BGR', 'BO': 'BOL', 'BR': 'BRA', 'BY': 'BLR', 'CA': 'CAN',
    'CH': 'CHE', 'CL': 'CHL', 'CO': 'COL', 'CR': 'CRI', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'DO': 'DOM', 'EC': 'ECU', 'EE': 'EST',
    'EG': 'EGY', 'ES': 'ESP', 'FI': 'FIN', 'FR': 'FRA', 'GB': 'GBR',
    'GR': 'GRC', 'GT': 'GTM', 'HK': 'HKG', 'HN': 'HND', 'HU': 'HUN',
    'ID': 'IDN', 'IE': 'IRL', 'IL': 'ISR', 'IN': 'IND', 'IS': 'ISL',
    'IT': 'ITA', 'JP': 'JPN', 'KR': 'KOR', 'KZ': 'KAZ', 'LT': 'LTU',
    'LU': 'LUX', 'LV': 'LVA', 'MA': 'MAR', 'MX': 'MEX', 'MY': 'MYS',
    'NG': 'NGA', 'NI': 'NIC', 'NL': 'NLD', 'NO': 'NOR', 'NZ': 'NZL',
    'PA': 'PAN', 'PE': 'PER', 'PH': 'PHL', 'PK': 'PAK', 'PL': 'POL',
    'PT': 'PRT', 'PY': 'PRY', 'RO': 'ROU', 'SA': 'SAU', 'SE': 'SWE',
    'SG': 'SGP', 'SK': 'SVK', 'SV': 'SLV', 'TH': 'THA', 'TR': 'TUR',
    'TW': 'TWN', 'UA': 'UKR', 'US': 'USA', 'UY': 'URY', 'VE': 'VEN',
    'VN': 'VNM', 'ZA': 'ZAF'
}



# Function to map country code to full country name
def map_country_code_to_name(country_code):
    return pycountry.countries.get(alpha_3=country_code).name

# # Example usage
# full_country_name = map_country_code_to_name('USA')
# print(full_country_name)  # Output: United States



# Make a copy to preserve the original dataframe
spotify_data_countries_copy = spotify_data_countries.copy()

# Replace country codes with ISO Alpha-3 codes in your DataFrame
spotify_data_countries_copy['country'] = spotify_data_countries_copy['country'].map(country_mapping)

# Filter out rows with missing or unmapped country codes
spotify_data_countries_copy = spotify_data_countries_copy.dropna(subset=['country'])

# Convert snapshot_date to datetime
spotify_data_countries_copy['snapshot_date'] = pd.to_datetime(spotify_data_countries_copy['snapshot_date'])


# Define layout of tab 3 (COUNTRIES, GLOBAL)
tab3_content = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Step1: Select the Range of Popularity (mainstream or undiscovered)", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),  # Light green background, bold, and centered text
                dcc.RangeSlider(
                    id='color-scale-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=[0, 50],
                    marks={i: str(i) for i in range(0, 101, 5)},
            )], color="light"),
            dbc.Card([
                dbc.CardHeader("Step2: Zoom in & Select a Country to View the Top 10 Songs", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),  # Light green background, bold, and centered text
                dcc.Graph(
                    id='choropleth-map',
                    style={'height': '100%', 'padding': '3px'} # Adjusted height 58vh was good
                )
            ], color="light", style={'margin-top': '16px'}) # style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px', 'margin-top': '16px'}
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Most Frequently Ranked Artists in the World by Popularity", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                dcc.Graph(
                    id='top-artists-pie-chart',
                    config={'displayModeBar': False}, # Hide the mode bar
                    style={'height': '100%', 'padding': '3px'} # Adjusted height 35vh was good, but chart wouldn't fit
                ),
            ], color="light" # , style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px'}
            ),
            dbc.Card([
                dbc.CardHeader("Top 3 Artists in the World based on Popularity Range", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),
                html.Div(id='image-container', style={'height': '33vh', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}) # height was 27vh
            ], color="light", style={'margin-top': '16px'} # 'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 
            )
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                html.Div(id='selected-country'),
                html.Div(id='song-list')
            ], color="light", style={'margin-top': '16px'}) # 'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 
        ], width=12),
    ]),
], fluid=True)

# Define callback to update choropleth map based on slider value range
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('color-scale-slider', 'value')]
)
def update_choropleth_map(selected_range):
    # Extracting min and max values from the selected range
    min_value, max_value = selected_range

    # Filter DataFrame based on selected range of popularity values
    filtered_data = spotify_data_countries_copy[(spotify_data_countries_copy['popularity'] >= min_value) & 
                                                (spotify_data_countries_copy['popularity'] <= max_value)]

    # Sort filtered data by 'popularity'
    filtered_data = filtered_data.sort_values(by='popularity', ascending=False)

    # Group by country and find the most recent snapshot for each country
    most_recent_data = filtered_data.groupby('country').apply(lambda x: x.loc[x['snapshot_date'].idxmax()]).reset_index(drop=True)

    # Create hover text with name, artists, and popularity for filtered data
    hover_text_filtered = most_recent_data['name'] + ' by ' + most_recent_data['artists'] + '<br>' + \
                          'Popularity: ' + most_recent_data['popularity'].astype(str) + '<br>' + \
                          'Country: ' + most_recent_data['country']

    # Create choropleth map figure for filtered data
    fig = go.Figure(go.Choropleth(
        locations=most_recent_data['country'],
        z=most_recent_data['popularity'],
        locationmode='ISO-3',
        colorscale='Viridis', # 'Cividis', # 'Inferno', # 'Viridis', # 'Plasma', # ['red', 'yellow', 'green']
        colorbar_title='Popularity',
        hovertext=hover_text_filtered,
        hoverinfo='text',
        zmin=0,
        zmax=100,
        showscale=True,
    ))

    # Customize the layout of the choropleth map
    fig.update_layout(
        # title='Top Songs by Country (Based on Popularity)',
        geo=dict(
            showcoastlines=True,
            coastlinecolor="DarkBlue",
            showland=True,
            landcolor="LightGrey",
            showocean=True,
            oceancolor="LightBlue"
        )
    )

    return fig

# # Add callback to update bar chart of the count of the top 10 most frequent song names globally
# @app.callback(
#     Output('top-songs-bar-chart', 'figure'),
#     [Input('color-scale-slider', 'value')]
# )
# def update_top_songs_bar_chart(selected_range):
#     # Extracting min and max values from the selected range
#     min_value, max_value = selected_range

#     # Filter DataFrame based on selected range of popularity values
#     filtered_data = spotify_data_countries_copy[(spotify_data_countries_copy['popularity'] >= min_value) & 
#                                                 (spotify_data_countries_copy['popularity'] <= max_value)]

#     # Count the occurrences of each song name
#     top_song_counts = filtered_data['name'].value_counts().head(10)
    
#     # Get the corresponding artists for the top songs
#     top_song_artists = filtered_data.groupby('name')['artists'].first()
    
#     # Create hover text with name and artists
#     hover_text = [f"{song}<br>by {top_song_artists[song]}" for song in top_song_counts.index]
    
#     # Create a horizontal bar chart
#     fig = go.Figure(data=[go.Bar(
#         y=top_song_counts.index + ' by ' + top_song_artists[top_song_counts.index], # Concatenate song name and artist
#         x=top_song_counts.values,
#         orientation='h', # Set orientation to horizontal
#         hovertext=hover_text,
#         hoverinfo='text',
#     )])
#     fig.update_layout(#title='Top 10 Most Frequently Ranked Songs by Popularity (Globally)', 
#                       yaxis={'categoryorder': 'total ascending'},
#                       xaxis={'side': 'top'}, # Move x-axis markings to the top
#                       font=dict(size=10))
        
#     return fig

# Add callback to update top artists pie chart based on slider value
@app.callback(
    Output('top-artists-pie-chart', 'figure'),
    [Input('color-scale-slider', 'value')]
)
def update_top_artists_pie_chart(selected_range):
    # Extracting min and max values from the selected range
    min_value, max_value = selected_range

    # Filter DataFrame based on selected range of popularity values
    filtered_data = spotify_data_countries_copy[(spotify_data_countries_copy['popularity'] >= min_value) & 
                                                (spotify_data_countries_copy['popularity'] <= max_value)]
    
    # Count the occurrences of each artist
    top_artist_counts = filtered_data['artists'].value_counts().head(10)
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=top_artist_counts.index,
        values=top_artist_counts.values,
    )])
        
    return fig

# Add callback to update top artists bar chart based on slider value
@app.callback(
    Output('image-container', 'children'),
    [Input('color-scale-slider', 'value')]
)
def update_top_artists_img(selected_range):
    # Extracting min and max values from the selected range
    min_value, max_value = selected_range

    # Filter DataFrame based on selected range of popularity values
    filtered_data = spotify_data_countries_copy[(spotify_data_countries_copy['popularity'] >= min_value) & 
                                                (spotify_data_countries_copy['popularity'] <= max_value)]
    

    # Count the occurrences of each artist
#     filtered_data_notdup = filtered_data[['artists', 'country']].drop_duplicates()
#     list_top_artists = filtered_data_notdup['artists'].value_counts().head(3).index.tolist()
    list_top_artists = filtered_data['artists'].value_counts().head(3).index.tolist()
    
#     list_top_artists = ['Taylor Swift',  'Justin Bieber', 'Ed Sheeran']  # top 3

    list_links_picts = []
    image_components = []
    for i, search_artist in enumerate(list_top_artists):
        # define website (Bing is easy for scraping)
        # url_search = f'https://www.bing.com/images/search?q={search_query}'
        url_search = f'https://www.bing.com/images/search?cw=1853&ch=933&q=spotify+{search_artist}&qft=%2bfilterui%3aface-portrait&first=1'
        url_search

        # call html
        headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
            }

        response = requests.get(url_search, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        time.sleep(1.5)  # wait 1 second for data showing up; I have tried 0.75. It cannot load the image.

        # get image link
        link_image = soup.find_all('img', {"class":"mimg"})[0].get('src')

        image_components.append(
            html.Div([  # Create a container div for image and name
                html.Img(src=link_image, style={'height': '180px', 'width': '170px', 'margin': '5px'}),
                html.P(search_artist)  # Add a paragraph for the artist's name
            ], style={'display': 'inline-block', 'margin': '5px', 'text-align': 'center'})  # Style for spacing
        )
    
    
    return image_components


# Define callback to update selected country display
@app.callback(
    Output('selected-country', 'children'),
    [Input('choropleth-map', 'clickData')]
)
def update_selected_country_display(clickData):
    if clickData:
        country_code = clickData['points'][0]['location']
        country_name = map_country_code_to_name(country_code) # Map country code to full country name
        return html.H3(f"Selected Country: {country_name}")

    return html.H3("Click on a country to see its top 10 songs by popularity.")

# Add callback to update song list when a country is clicked or slider value changes
@app.callback(
    Output('song-list', 'children'),
    [Input('choropleth-map', 'clickData'),
     Input('color-scale-slider', 'value')]
)
def update_song_list(clickData, selected_range):
    # if not clickData:
    #     return "Click on a country to see its top 10 songs by popularity."

    if clickData:

        country_clicked = clickData['points'][0]['location']
        top_songs_in_country = spotify_data_countries_copy[spotify_data_countries_copy['country'] == country_clicked]

        # Filter songs based on popularity less than or equal to the selected value range
        min_value, max_value = selected_range
        top_songs_filtered = top_songs_in_country[(top_songs_in_country['popularity'] >= min_value) & 
                                                    (top_songs_in_country['popularity'] <= max_value)]

        # Drop duplicates based on name and artists to keep only one entry for each song
        top_songs_unique = top_songs_filtered.drop_duplicates(subset=['name', 'artists'])

        # Select the top 10 songs by popularity after removing duplicates
        top_songs_top10 = top_songs_unique.nlargest(10, 'popularity')

        # Additional columns to include with each first letter capitalized in the header
        columns = ['popularity', 
#                    'danceability', 'energy', 'loudness', 'speechiness',
#                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
                  ]

        # Create DataTable component for displaying top 10 songs with additional columns
        data_table = dash_table.DataTable(
            id='table',
            columns=[{'name': col.capitalize(), 'id': col} for col in ['name', 'artists'] + columns],
            data=top_songs_top10.to_dict('records'),
            style_cell={'textAlign': 'left'}
        )

        return data_table



# ## Merge Tab
# Define the app layout
app.layout = dbc.Container([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab([tab1_content], label='Discover Music Taste', value='tab-1',
                style={'fontSize': 20, 'color': 'black', },
                selected_style={'fontSize': 20, 'fontWeight': 'bold', 'color': 'white', 'backgroundColor': 'green'}),
        dcc.Tab([tab2_content], label='Find New One?', value='tab-2',
                style={'fontSize': 20, 'color': 'black'},
                selected_style={'fontSize': 20, 'fontWeight': 'bold', 'color': 'white', 'backgroundColor': 'green'}),
        dcc.Tab([tab3_content], label='Explore the World', value='tab-3',
                style={'fontSize': 20, 'color': 'black'},
                selected_style={'fontSize': 20, 'fontWeight': 'bold', 'color': 'white', 'backgroundColor': 'green'}),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=False)