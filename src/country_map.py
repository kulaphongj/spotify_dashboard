#!/usr/bin/env python

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from dash import dash_table
import pandas as pd

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

# Make a copy to preserve the original dataframe
spotify_data_countries_copy = spotify_data_countries.copy()

# Replace country codes with ISO Alpha-3 codes in your DataFrame
spotify_data_countries_copy['country'] = spotify_data_countries_copy['country'].map(country_mapping)

# Filter out rows with missing or unmapped country codes
spotify_data_countries_copy = spotify_data_countries_copy.dropna(subset=['country'])

# Convert snapshot_date to datetime
spotify_data_countries_copy['snapshot_date'] = pd.to_datetime(spotify_data_countries_copy['snapshot_date'])

# Create Dash app
app = dash.Dash(__name__)

# Define layout of the app
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='choropleth-map'),
            dcc.Slider(
                id='color-scale-slider',
                min=0,
                max=100,
                step=1,
                value=50,
                marks={i: str(i) for i in range(0, 101, 5)},
            ),
            html.Div(id='selected-country')
        ], style={'width': '40%', 'float': 'left'}),
        
        html.Div([
            dcc.Graph(id='top-songs-bar-chart')
        ], style={'width': '40%', 'float': 'right', 'display': 'inline-block', 'margin-top': '10px'})
    ]),
    
    html.Div([
        html.Div(id='song-list')
    ], style={'width': '100%', 'float': 'left'})
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

# Add callback to update selected country display
@app.callback(
    Output('selected-country', 'children'),
    [Input('choropleth-map', 'clickData')]
)
def update_selected_country_display(clickData):
    if clickData:
        country_clicked = clickData['points'][0]['location']
        return html.H3(f"Selected Country: {country_clicked}")

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
            style_cell={'textAlign': 'left'} # Align cell text to the left
        )

        return data_table

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
    
    # Create a bar chart
    fig = go.Figure(data=[go.Bar(
        x=top_song_counts.index + ' by ' + top_song_artists[top_song_counts.index], # Concatenate song name and artist
        y=top_song_counts.values,
        hovertext=hover_text,
        hoverinfo='text',
    )])
    fig.update_layout(title='Top 10 Most Frequently Ranked Songs by Popularity (Globally)')
        
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    