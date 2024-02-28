from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc  # Import Bootstrap components
import plotly.graph_objects as go



# FOR FIRST TAB -----------------------------------------------------------
# Define the datasets or load them if not defined
spotify_data_genres = pd.read_csv("Datasets/spotify2/train.csv", encoding='ISO-8859-1')
unique_genres = set(spotify_data_genres['track_genre'])

# Specify the columns for the radar chart
columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
           'instrumentalness', 'liveness', 'valence', 'tempo']

# FOR SECOND TAB -----------------------------------------------------------
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

# Load Spotify data and preprocess it
spotify_data_countries = pd.read_csv("Datasets/spotify/universal_top_spotify_songs.csv")
spotify_data_countries_copy = spotify_data_countries.copy()
spotify_data_countries_copy['country'] = spotify_data_countries_copy['country'].map(country_mapping)
spotify_data_countries_copy['snapshot_date'] = pd.to_datetime(spotify_data_countries_copy['snapshot_date'])  # Convert to datetime
spotify_data_countries_copy = spotify_data_countries_copy.dropna(subset=['country'])



# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



# Define the first tab layout
tab1_layout = html.Div([
    html.H1('Genre Radar Chart'),
    dcc.Graph(id='radar-chart'),
    dcc.Dropdown(
        id='dropdown-menu',
        options=[
            {'label': 'Select all', 'value': 'all'},  # Select all option
        ] + [{'label': genre, 'value': genre} for genre in unique_genres],
        multi=True,
        value=list(unique_genres)[:10]  # Default value, selecting first 10 genres
    ),
])

# Define the second tab layout
tab2_layout = html.Div([
    dcc.Graph(id='choropleth-map'),
    dcc.Slider(
        id='color-scale-slider',
        min=0,
        max=100,
        step=1,
        value=50,
        marks={i: str(i) for i in range(0, 101, 5)},
    ),
    # # Load Spotify data and preprocess it
    # dcc.Store(id='spotify-data-store', data=spotify_data_countries_copy.to_dict('records')),
])

# Define the tabs
app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label="Genre Radar Chart", children=tab1_layout),
        dcc.Tab(label="Top Songs by Country", children=tab2_layout),
    ]),
])



# Define callback to update radar chart based on dropdown menu value
@app.callback(
    Output('radar-chart', 'figure'),
    [Input('dropdown-menu', 'value')]
)
def update_radar_chart(selected_genres):
    if 'all' in selected_genres:
        selected_genres = list(unique_genres)

    # Group by genre and calculate the mean of each metric
    mean_metrics_by_genre = spotify_data_genres.groupby('track_genre')[columns].mean()

    # Filter the mean metrics data for the selected genres
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
            theta=columns,
            fill='toself',
            name=genre
        ))

    # Create the layout
    layout = go.Layout(
        title="Average Music Metrics by Genre (Radar Chart)",
        polar=dict(
            radialaxis=dict(visible=True),
        ),
        showlegend=True
    )

    # Create the radar chart figure
    fig = go.Figure(data=data, layout=layout)

    return fig



# Define callback to update choropleth map based on slider value
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('color-scale-slider', 'value')] # ,
    #  Input('spotify-data-store', 'data')]  # Include input for loaded data
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
        colorscale='Plasma',  # Adjust color scale as needed
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



if __name__ == '__main__':
    app.run_server(debug=True)
    
    
