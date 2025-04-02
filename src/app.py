import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.graph_objects as go  # Plotly Graph Objects for visualizations

# Load the dataset
data_path = "Solar_Orbiter_with_anomalies.csv"
data_path2 = "Solar_Orbiter_with_anomalies2.csv"
solar_data = pd.read_csv(data_path)
solar_data2 = pd.read_csv(data_path2)

# Convert the 'Date' column to datetime format
solar_data['Date'] = pd.to_datetime(solar_data['Date'])
solar_data2['Date'] = pd.to_datetime(solar_data2['Date'])

# Filter out data from May 5 to May 11
start_exclude = pd.to_datetime('2021-05-05')
end_exclude = pd.to_datetime('2021-05-11')
solar_data = solar_data[~((solar_data['Date'] >= start_exclude) & (solar_data['Date'] <= end_exclude))]
solar_data2 = solar_data2[~((solar_data2['Date'] >= start_exclude) & (solar_data2['Date'] <= end_exclude))]

# Initialize the Dash app
app = dash.Dash(__name__, title="Solar Orbiter Data Visualization")
server = app.server

# Remove the 'Date' and 'anomaly_score' columns from the checklist options
checklist_options = sorted(
    [{'label': col, 'value': col} for col in solar_data.columns if col not in ['Date', 'anomaly_score']],
    key=lambda x: x['label']
)

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
        html.Div([dcc.Graph(id='time-series-chart')], className="six columns"),  # Time Series Chart
        html.Div([dcc.Graph(id='anomaly-score-chart')], className="six columns"),  # Anomaly Score Chart
    ], className="row")
])

# Callbacks to update graphs
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('anomaly-score-chart', 'figure')],
    [Input('instrument-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_instruments, start_date, end_date):
    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]
    filtered_data2 = solar_data2[(solar_data2['Date'] >= start_date) & (solar_data2['Date'] <= end_date)]

    # Time Series Chart
    time_series_fig = go.Figure()
    for instrument in selected_instruments:
        time_series_fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data[instrument],
                mode='lines+markers',
                name=instrument
            )
        )
    time_series_fig.update_layout(
        title="Time Series of Selected Instruments",
        xaxis_title="Date",
        yaxis_title="Instrument Value"
    )

    # Anomaly Score Chart
    anomaly_score_fig = go.Figure()
    anomaly_score_fig.add_trace(go.Scatter(
        x=filtered_data2['Date'],
        y=filtered_data2['anomaly_score'],
        mode='lines+markers',
        name='Anomaly Score',
        marker=dict(
            color=['red' if val < 0 else 'blue' for val in filtered_data2['anomaly_score']],
            size=5
        )
    ))
    anomaly_score_fig.update_layout(
        title="Anomaly Scores Over Time",
        xaxis_title='Date',
        yaxis_title='Anomaly Score'
    )

    return time_series_fig, anomaly_score_fig

if __name__ == "__main__":
    app.run_server(debug=True)
