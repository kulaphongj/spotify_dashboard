import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# Assuming 'df' is your DataFrame loaded from 'data.csv'
# Make sure to load your DataFrame here
df = pd.read_csv('data.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Music Dashboard", className="text-white"), style={'backgroundColor': 'steelblue', 'borderRadius': '3', 'padding': '15px', 'marginTop': '20px', 'marginBottom': '20px', 'marginRight': '15px'})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.Div(id='sliders-container', children=[
                        html.Div([
                            html.Label(f"{feature.capitalize()}"),
                            dcc.RangeSlider(
                                id=f'{feature}-slider',
                                min=df[feature].min() if df[feature].dtype != 'O' else 0,
                                max=df[feature].max() if df[feature].dtype != 'O' else 1,
                                step=0.01 if df[feature].dtype in [float, int] else 1,
                                marks=generate_marks(df[feature].min(), df[feature].max()),
                                value=[df[feature].min(), df[feature].max()]
                            ),
                        ], style={'padding': '10px', 'margin': '10px 0', 'borderRadius': '5px', 'background': '#f9f9f9'}) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                    ]),
                    dcc.Dropdown(
                        id='genre-dropdown',
                        options=[{'label': genre, 'value': genre} for genre in df['track_genre'].unique()],
                        value=df['track_genre'].unique()[0],
                        multi=True
                    ),
                    html.Label('Switch between Popular and Not Popular', style={'display': 'block', 'marginTop': '20px'}),
                    dcc.Checklist(
                        id='popularity-switch',
                        options=[{'label': ' Popular / Not Popular', 'value': 'ON'}],
                        value=[]
                    )
                ])
            ], style={'marginBottom': '20px'}),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Features Bar Chart"),
                dbc.CardBody([
                    dcc.Graph(id='features-bar-chart')
                ])
            ], style={'marginBottom': '20px'}),
            dbc.Card([
                dbc.CardHeader("Parallel Coordinates Plot"),
                dbc.CardBody([
                    dcc.Graph(id='parallel-coordinates-plot')
                ])
            ], style={'marginBottom': '20px'}),
            dbc.Card([
                dbc.CardHeader("Songs Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='songs-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        page_size=10,
                        style_header={'fontWeight': 'bold', 'textAlign': 'left'},
                        style_cell={'textAlign': 'left'},
                    )
                ])
            ])
        ], md=8)
    ])
])




@app.callback(
    [Output('features-bar-chart', 'figure'),
     Output('parallel-coordinates-plot', 'figure'),
     Output('songs-table', 'data')],
    [Input(f'{feature}-slider', 'value') for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']] +
    [Input('genre-dropdown', 'value')]
)
def update_content(*args):
    slider_values = args[:-1]  # All slider values
    selected_genres = args[-1]  # Genre dropdown value

    # Check if selected_genres is a string and convert to list if necessary
    filtered_df = df[df['track_genre'].isin([selected_genres])] if isinstance(selected_genres, str) else df[df['track_genre'].isin(selected_genres)]

    for i, feature in enumerate(['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        min_val, max_val = slider_values[i]
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

    normalized_df = normalize(filtered_df, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

    features_means = filtered_df[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].mean()
    bar_chart = go.Figure(data=[go.Bar(x=features_means.index, y=features_means.values, marker=dict(color='rgba(50, 171, 96, 0.7)', line=dict(color='rgba(50, 171, 96, 1.0)', width=2)))])
    bar_chart.update_layout(title_text='Musical Features Proportions', xaxis_title='Musical Feature', yaxis_title='Average', yaxis=dict(range=[0, 1]))

    parallel_coordinates_fig = go.Figure(data=go.Parcoords(line=dict(color=normalized_df['popularity'], colorscale='Electric', showscale=True), dimensions=[dict(range=[0, 1], label=feature.capitalize(), values=normalized_df[feature]) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]))
    parallel_coordinates_fig.update_layout(title_text='Parallel Coordinates Plot of Musical Features')

    table_data = filtered_df.to_dict('records')

    return bar_chart, parallel_coordinates_fig, table_data



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) as go
import pandas as pd
import dash_bootstrap_components as dbc

# Assuming 'df' is your DataFrame loaded from 'data.csv'
# Make sure to load your DataFrame here
df = pd.read_csv('data.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Music Dashboard", className="text-white"), style={'backgroundColor': 'steelblue', 'borderRadius': '3', 'padding': '15px', 'marginTop': '20px', 'marginBottom': '20px', 'marginRight': '15px'})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.Div(id='sliders-container', children=[
                        html.Div([
                            html.Label(f"{feature.capitalize()}"),
                            dcc.RangeSlider(
                                id=f'{feature}-slider',
                                min=df[feature].min() if df[feature].dtype != 'O' else 0,
                                max=df[feature].max() if df[feature].dtype != 'O' else 1,
                                step=0.01 if df[feature].dtype in [float, int] else 1,
                                marks=generate_marks(df[feature].min(), df[feature].max()),
                                value=[df[feature].min(), df[feature].max()]
                            ),
                        ], style={'padding': '10px', 'margin': '10px 0', 'borderRadius': '5px', 'background': '#f9f9f9'}) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                    ]),
                    dcc.Dropdown(
                        id='genre-dropdown',
                        options=[{'label': genre, 'value': genre} for genre in df['track_genre'].unique()],
                        value=df['track_genre'].unique()[0],
                        multi=True
                    ),
                    html.Label('Switch between Popular and Not Popular', style={'display': 'block', 'marginTop': '20px'}),
                    dcc.Checklist(
                        id='popularity-switch',
                        options=[{'label': ' Popular / Not Popular', 'value': 'ON'}],
                        value=[]
                    )
                ])
            ], style={'marginBottom': '20px'}),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Features Bar Chart"),
                dbc.CardBody([
                    dcc.Graph(id='features-bar-chart')
                ])
            ], style={'marginBottom': '20px'}),
            dbc.Card([
                dbc.CardHeader("Parallel Coordinates Plot"),
                dbc.CardBody([
                    dcc.Graph(id='parallel-coordinates-plot')
                ])
            ], style={'marginBottom': '20px'}),
            dbc.Card([
                dbc.CardHeader("Songs Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='songs-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        page_size=10,
                        style_header={'fontWeight': 'bold', 'textAlign': 'left'},
                        style_cell={'textAlign': 'left'},
                    )
                ])
            ])
        ], md=8)
    ])
])




@app.callback(
    [Output('features-bar-chart', 'figure'),
     Output('parallel-coordinates-plot', 'figure'),
     Output('songs-table', 'data')],
    [Input(f'{feature}-slider', 'value') for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']] +
    [Input('genre-dropdown', 'value')]
)
def update_content(*args):
    slider_values = args[:-1]  # All slider values
    selected_genres = args[-1]  # Genre dropdown value

    # Check if selected_genres is a string and convert to list if necessary
    filtered_df = df[df['track_genre'].isin([selected_genres])] if isinstance(selected_genres, str) else df[df['track_genre'].isin(selected_genres)]

    for i, feature in enumerate(['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        min_val, max_val = slider_values[i]
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

    normalized_df = normalize(filtered_df, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

    features_means = filtered_df[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].mean()
    bar_chart = go.Figure(data=[go.Bar(x=features_means.index, y=features_means.values, marker=dict(color='rgba(50, 171, 96, 0.7)', line=dict(color='rgba(50, 171, 96, 1.0)', width=2)))])
    bar_chart.update_layout(title_text='Musical Features Proportions', xaxis_title='Musical Feature', yaxis_title='Average', yaxis=dict(range=[0, 1]))

    parallel_coordinates_fig = go.Figure(data=go.Parcoords(line=dict(color=normalized_df['popularity'], colorscale='Electric', showscale=True), dimensions=[dict(range=[0, 1], label=feature.capitalize(), values=normalized_df[feature]) for feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]))
    parallel_coordinates_fig.update_layout(title_text='Parallel Coordinates Plot of Musical Features')

    table_data = filtered_df.to_dict('records')

    return bar_chart, parallel_coordinates_fig, table_data



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
