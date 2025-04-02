import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more control over visualizations
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_path = "Solar_Orbiter_with_anomalies.csv"  # Path to dataset file
data_path2 = "Solar_Orbiter_with_anomalies2.csv"
solar_data = pd.read_csv(data_path)  # Read dataset into DataFrame
solar_data2 = pd.read_csv(data_path2)

# Convert the 'Date' column to datetime format
solar_data['Date'] = pd.to_datetime(solar_data['Date'])
solar_data2['Date'] = pd.to_datetime(solar_data2['Date'])

# Filter out data from May 5 to May 11
start_exclude = pd.to_datetime('2021-05-05')
end_exclude = pd.to_datetime('2021-05-11')
solar_data = solar_data[~((solar_data['Date'] >= start_exclude) & (solar_data['Date'] <= end_exclude))]
solar_data2 = solar_data2[~((solar_data2['Date'] >= start_exclude) & (solar_data2['Date'] <= end_exclude))]

# Load the SHAP values data
shap_values_path = "shap_values.csv"
shap_data = pd.read_csv(shap_values_path)

# Create the feature importance figure
feature_importance_fig = px.line(shap_data, x='Date', y=shap_data.columns[1:],
                                 title='Feature Importance for Predicting Anomalies On Different Dates',
                                 labels={'value': 'SHAP Value', 'Date': 'Date'},
                                 template='plotly')

# Initialize the Dash app
app = dash.Dash(__name__, title="Solar Orbiter Data Visualization")
server = app.server

# Remove the 'Date' and 'anomaly_score' columns from the checklist options
checklist_options = [{'label': col, 'value': col} for col in solar_data.columns if col not in ['Date', 'anomaly_score']]

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Solar Orbiter Instrument Data Visualization", style={'text-align': 'center'}),
    dcc.Checklist(
        id='instrument-checklist',
        options=checklist_options,
        value=[solar_data.columns[1]],
        inline=True
    ),
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=solar_data['Date'].min(),
        max_date_allowed=solar_data['Date'].max(),
        start_date=solar_data['Date'].min(),
        end_date=solar_data['Date'].max()
    ),
    html.Div([
        dcc.Graph(id='time-series-chart'),
        dcc.Graph(id='correlation-heatmap')
    ], className="row"),
    html.Div([
        dcc.Graph(id='scaled-time-series-chart'),
        dcc.Graph(id='anomaly-score-chart')
    ], className="row"),
    html.Div([dcc.Graph(figure=feature_importance_fig, id='feature-importance-chart')])
])

@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('scaled-time-series-chart', 'figure'),
     Output('anomaly-score-chart', 'figure')],
    [Input('instrument-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_instruments, start_date, end_date):
    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]
    filtered_data2 = solar_data2[(solar_data2['Date'] >= start_date) & (solar_data2['Date'] <= end_date)]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = filtered_data.copy()
    scaled_data[selected_instruments] = scaler.fit_transform(filtered_data[selected_instruments])

    time_series_fig = go.Figure()
    for instrument in selected_instruments:
        time_series_fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data[instrument], mode='lines', name=instrument))

    correlation_fig = go.Figure(go.Heatmap(z=scaled_data[selected_instruments].corr(), x=selected_instruments, y=selected_instruments, colorscale='Viridis'))

    scaled_time_series_fig = go.Figure()
    for instrument in selected_instruments:
        scaled_time_series_fig.add_trace(go.Scatter(x=scaled_data['Date'], y=scaled_data[instrument], mode='lines', name=instrument))

    anomaly_score_fig = go.Figure()
    anomaly_score_fig.add_trace(go.Scatter(x=filtered_data2['Date'], y=filtered_data2['anomaly_score'], mode='lines', name='Anomaly Score'))
    
    return time_series_fig, correlation_fig, scaled_time_series_fig, anomaly_score_fig

if __name__ == "__main__":
    app.run_server(debug=True)
