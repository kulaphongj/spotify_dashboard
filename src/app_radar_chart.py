import dash
from dash import dcc, html
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define the datasets or load them if not defined
spotify_data_genres = pd.read_csv("Datasets/spotify2/train.csv", encoding='ISO-8859-1')
unique_genres = set(spotify_data_genres['track_genre'])

# Specify the columns for the radar chart
columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
           'instrumentalness', 'liveness', 'valence', 'tempo']

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
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

# Define the callback function
@app.callback(
    dash.dependencies.Output('radar-chart', 'figure'),
    [dash.dependencies.Input('dropdown-menu', 'value')]
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

if __name__ == '__main__':
    app.run_server(debug=True)
