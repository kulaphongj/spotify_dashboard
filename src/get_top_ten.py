import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Load your DataFrame
df = pd.read_csv('data.csv')

app = dash.Dash(__name__)

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


app.layout = html.Div([
    html.H1("Music Dashboard"),
    # Left side: Filters
    html.Div([
        html.Div(id='sliders-container', children=[
            html.Div([
                html.Label(f"{feature.capitalize()}"),
                dcc.RangeSlider(
                    id=f'{feature}-slider',
                    min=df[feature].min() if df[feature].dtype != bool else 0,
                    max=df[feature].max() if df[feature].dtype != bool else 1,
                    step=0.01 if df[feature].dtype in [float, int] else 1,
                    marks=generate_marks(df[feature].min(), df[feature].max()),
                    value=[df[feature].min(), df[feature].max()]
                ),
            ],  style={'padding': '20px', 'margin': '10px 0', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'background': '#f9f9f9'}) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        ]),
        dcc.Dropdown(
            id='genre-dropdown',
            options=[{'label': genre, 'value': genre} for genre in df['track_genre'].unique()],
            value=[df['track_genre'].unique()[0]],
            multi=True
        ),
        html.Label('Switch between Popular and Not Popular', style={'display': 'block', 'marginTop': '20px'}),
        dcc.Checklist(
            id='popularity-switch',
            options=[{'label': ' Popular / Not Popular', 'value': 'ON'}],
            value=[]
        )

    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

    # Right side: Visualizations
    html.Div([
        dcc.Graph(id='features-bar-chart', style={'display': 'inline-block', 'width': '30%'}),

        dcc.Graph(id='parallel-coordinates-plot', style={'display': 'inline-block', 'width': '70%'}),  # Parallel Coordinates Plot placeholder
        dash_table.DataTable(
            id='songs-table',
            columns=[
                {"name": "Artists", "id": "artists"},
                {"name": "Album Name", "id": "album_name"},
                {"name": "Track Name", "id": "track_name"},
                {"name": "Popularity", "id": "popularity"}
            ],
            style_table={'height': '300px', 'overflowY': 'auto'},
            page_size=10,
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'left',
            },
            style_cell={
                'textAlign': 'left',
            },
        ),
    ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
])

@app.callback(
    [Output('features-bar-chart', 'figure'),  # Update the ID here
     Output('parallel-coordinates-plot', 'figure'),
     Output('songs-table', 'data')],
    [Input(f'{feature}-slider', 'value') for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']] +
    [Input('genre-dropdown', 'value')]
)

def update_content(*args):
    slider_values = args[:-1]  # All slider values
    selected_genres = args[-1]  # Genre dropdown value

    # Filter data
    filtered_df = df[df['track_genre'].isin(selected_genres)]
    for i, feature in enumerate(['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        filtered_df = filtered_df[(filtered_df[feature] >= slider_values[i][0]) & (filtered_df[feature] <= slider_values[i][1])]

    # Normalize the filtered data
    features_to_normalize = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    normalized_df = normalize(filtered_df, features_to_normalize)

    # Calculate the average of each musical feature
    features_means = filtered_df[features_to_normalize].mean()

    # Generate the vertical bar chart for the average of each musical feature
    bar_chart = go.Figure(data=[
        go.Bar(
            x=features_means.index,
            y=features_means,
            marker=dict(color='rgba(50, 171, 96, 0.7)', line=dict(color='rgba(50, 171, 96, 1.0)', width=2))
        )
    ])
    bar_chart.update_layout(title_text='Musical Features Proportions', xaxis_title='Musical Feature', yaxis_title='Average', yaxis=dict(range=[0, 1]))

    # Generate the Parallel Coordinates Plot
    parallel_coordinates_fig = go.Figure(data=go.Parcoords(
        line=dict(color=normalized_df['popularity'], colorscale='Electric', showscale=True),
        dimensions=[dict(range=[0, 1], label=feature.capitalize(), values=normalized_df[feature]) for feature in features_to_normalize]
    ))
    parallel_coordinates_fig.update_layout(title_text='Parallel Coordinates Plot of Musical Features')

    # Update table data
    table_data = filtered_df.nlargest(10, 'popularity')[['artists', 'album_name', 'track_name', 'popularity']].to_dict('records')

    return bar_chart, parallel_coordinates_fig, table_data

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
