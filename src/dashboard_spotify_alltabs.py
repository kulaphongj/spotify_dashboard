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


# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# # Tabs

# ## Tab 1

# Data Preprocessing Tab1
# load data
df_tracks = pd.read_csv('./data/preprocessed/df_tracks_remove_dup.csv')

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
# list_artists = list_artists[:20]
# list_track_name = list_track_name[:5]

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
                            html.Iframe(
                                        id='pie-chart',
                                        style={'border-width': '0', 'width': '100%', 'height': '330px'}
                            )
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
    Output('pie-chart', 'srcDoc'),  # Genre Pie Chart
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
    
    
    chart_pie_t = (chart_pie + text).properties(width=230, height=270, 
                                                background='transparent').configure_view(strokeWidth=0)
    
    return df_table.to_dict('records'), chart_pie_t.to_html()


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

# Helper functions for generating slider marks and normalizing data
def generate_marks(feature_min, feature_max):
    step = max((feature_max - feature_min) / 5, 1)  # Ensure step is at least 1
    return {i: f"{i:.2f}" for i in range(int(feature_min), int(feature_max) + 1, int(step))}

def normalize(df, features):
    result = df.copy()
    for feature_name in features:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


tab2_content = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Discover New Music", className="text-center mb-4"), width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Music Features", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),  # Light green background, bold, and centered text
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Label(f"{feature.capitalize()}"),
                            dcc.RangeSlider(
                                id=f'{feature}-slider',
                                min=df[feature].min(),
                                max=df[feature].max(),
                                step=0.01,
                                marks=generate_marks(df[feature].min(), df[feature].max()),
                                value=[df[feature].min(), df[feature].max()],
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={'padding': '10px', 'margin': '10px 0'}) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                    ]),
                    dcc.Dropdown(
                        id='genre-dropdown',
                        options=[{'label': genre, 'value': genre} for genre in df['track_genre'].unique()],
                        value=[df['track_genre'].unique()[0]],
                        multi=True
                    ),
                ]),
            ], style={'marginBottom': '20px'}),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Selected Features and Trends", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),  # Light green background, bold, and centered text
                dbc.CardBody([
                    dcc.Graph(id='parallel-coordinates-plot'),
                ])
            ], style={'marginBottom': '20px'}),
            dbc.Card([
                dbc.CardHeader("Top 10 Songs", style={'backgroundColor': '#68A58C', 'fontWeight': 'bold', 'textAlign': 'center'}),  # Light green background, bold, and centered text
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
            ])
        ], md=8),
    ]),
])

# Callback for updating the Parallel Coordinates Plot and Songs Table based on user input
@app.callback(
    [Output('parallel-coordinates-plot', 'figure'),
     Output('songs-table', 'data')],
    [Input(f'{feature}-slider', 'value') for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']] +
    [Input('genre-dropdown', 'value')]
)
def update_content(*args):
    slider_values = args[:-1]  # Extract slider values
    selected_genres = args[-1]  # Extract genre dropdown value
    filtered_df = df[df['track_genre'].isin(selected_genres)]
    for i, feature in enumerate(['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        filtered_df = filtered_df[(filtered_df[feature] >= slider_values[i][0]) & (filtered_df[feature] <= slider_values[i][1])]
    
    # Normalize the filtered data for better visual comparison
    normalized_df = normalize(filtered_df, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

    # Parallel Coordinates Plot for visualizing the songs
    parallel_coordinates_figure = go.Figure(data=go.Parcoords(
        line=dict(color=normalized_df['popularity'], colorscale=[[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']], showscale=True),
        dimensions=[{'label': col, 'values': normalized_df[col]} for col in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    ))
    parallel_coordinates_figure.update_layout(title_text='Parallel Coordinates Plot for Selected Features')

    songs_table_data = filtered_df[['artists', 'track_id', 'track_name', 'popularity']].sort_values(by='popularity', ascending=False)
    songs_table_data['track_id'] = songs_table_data['track_id'].apply(lambda x: f"[Listen on Spotify](https://open.spotify.com/track/{x})")
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
tab3_content = html.Div([
    html.Div([
        html.Div([
            dbc.Card([
                dcc.Graph(
                    id='choropleth-map',
                    style={'height': '80vh'}
                           #'width': '790px'}, # '60vh'}, # Set height relative to the viewport height (60% of the viewport height)
                ),
                dcc.Slider(
                    id='color-scale-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=50,
                    marks={i: str(i) for i in range(0, 101, 5)},
                )
                # html.Div(id='selected-country') # Include selected-country below choropleth-map and slider
            ], style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px', 'margin-top': '0'})
        ], style={'width': '49%', 'float': 'left'}),
        
        html.Div([
            dbc.Card([
                dcc.Graph(
                    id='top-songs-bar-chart',
                    config={'displayModeBar': False}, # Hide the mode bar
                    style={'height': '40vh'}
                           # 'width': '790px'} # '30vh'} # Set height relative to the viewport height (30% of the viewport height)
                )
            ], style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px', 'marginTop': '0', 'width': '49%', 'float': 'right'})
        ]),
        
        html.Div([
            dbc.Card([
                dcc.Graph(id='top-artists-bar-chart',
                          config={'displayModeBar': False}, # Hide the mode bar
                          style={'height': '40vh'}
                                 # 'width': '790px'} # '30vh'} # Set height relative to the viewport height (30% of the viewport height)
                )
            ], style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px', 'marginTop': '3px', 'width': '49%', 'float': 'right', 'display': 'inline-block'})
        ]),
        
        # html.Div([
        #     dbc.Card([
        #         html.Div(id='selected-country')
        #     ], style={'backgroundColor': 'light', 'borderRadius': '5px', 'border': '1px solid lightgrey', 'padding': '5px', 'marginTop': '10px'})
        # ], style={'width': '100%', 'float': 'left'}),
        
        html.Div([
            dbc.Card([
                html.Div(id='selected-country'),
                html.Div(id='song-list')
            ], style={'backgroundColor': 'light', 'borderRadius': '10px', 'border': '1px solid lightgrey', 'padding': '3px', 'marginTop': '3px'})
        ], style={'width': '100%', 'float': 'left'}),
    ]),
])

# Define callback to update choropleth map based on slider value
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('color-scale-slider', 'value')]
)
def update_choropleth_map(selected_value):
    # Filter DataFrame based on selected popularity value
    filtered_data = spotify_data_countries_copy[spotify_data_countries_copy['popularity'] <= selected_value]

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
        title='Top Songs by Country (Based on Popularity)',
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

# Add callback to update bar chart of the count of the top 10 most frequent song names globally
@app.callback(
    Output('top-songs-bar-chart', 'figure'),
    [Input('color-scale-slider', 'value')]
)
def update_top_songs_bar_chart(selected_value):
    # Filter DataFrame based on selected popularity value
    filtered_data = spotify_data_countries_copy[spotify_data_countries_copy['popularity'] <= selected_value]

    # Count the occurrences of each song name
    top_song_counts = filtered_data['name'].value_counts().head(10)
    
    # Get the corresponding artists for the top songs
    top_song_artists = filtered_data.groupby('name')['artists'].first()
    
    # Create hover text with name and artists
    hover_text = [f"{song}<br>by {top_song_artists[song]}" for song in top_song_counts.index]
    
    # Create a horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        y=top_song_counts.index + ' by ' + top_song_artists[top_song_counts.index], # Concatenate song name and artist
        x=top_song_counts.values,
        orientation='h', # Set orientation to horizontal
        hovertext=hover_text,
        hoverinfo='text',
    )])
    fig.update_layout(title='Top 10 Most Frequently Ranked Songs by Popularity (Globally)', 
                      yaxis={'categoryorder': 'total ascending'},
                      xaxis={'side': 'top'}, # Move x-axis markings to the top
                      font=dict(size=10))
        
    return fig

# Add callback to update top artists bar chart based on slider value
@app.callback(
    Output('top-artists-bar-chart', 'figure'),
    [Input('color-scale-slider', 'value')]
)
def update_top_artists_bar_chart(selected_value):
    # Filter DataFrame based on selected popularity value
    filtered_data = spotify_data_countries_copy[spotify_data_countries_copy['popularity'] <= selected_value]

    # Count the occurrences of each artist
    top_artist_counts = filtered_data['artists'].value_counts().head(10)
    
    # Create a horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        y=top_artist_counts.index,
        x=top_artist_counts.values,
        orientation='h', # Set orientation to horizontal
    )])
    fig.update_layout(title='Top 10 Most Frequently Ranked Artists by Popularity (Globally)', 
                      yaxis={'categoryorder': 'total ascending'},
                      xaxis={'side': 'top'}, # Move x-axis markings to the top
                      font=dict(size=10))
        
    return fig

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
def update_song_list(clickData, selected_value):
    # if not clickData:
    #     return "Click on a country to see its top 10 songs by popularity."

    if clickData:

        country_clicked = clickData['points'][0]['location']
        top_songs_in_country = spotify_data_countries_copy[spotify_data_countries_copy['country'] == country_clicked]

        # Filter songs based on popularity less than or equal to the selected value
        top_songs_filtered = top_songs_in_country[top_songs_in_country['popularity'] <= selected_value]

        # Drop duplicates based on name and artists to keep only one entry for each song
        top_songs_unique = top_songs_filtered.drop_duplicates(subset=['name', 'artists'])

        # Select the top 10 songs by popularity after removing duplicates
        top_songs_top10 = top_songs_unique.nlargest(10, 'popularity')

        # Additional columns to include with each first letter capitalized in the header
        columns = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

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
        dcc.Tab([tab3_content], label='Explore Globally', value='tab-3',
                style={'fontSize': 20, 'color': 'black'},
                selected_style={'fontSize': 20, 'fontWeight': 'bold', 'color': 'white', 'backgroundColor': 'green'}),
    ])
])



if __name__ == '__main__':
    app.run_server(debug=True)


