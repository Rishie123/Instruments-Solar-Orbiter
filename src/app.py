import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html, dash_table  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more control over visualizations
from sklearn.preprocessing import MinMaxScaler

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

# Load the SHAP values data
shap_values_path = "shap_values.csv"
shap_data = pd.read_csv(shap_values_path)

# Create the feature importance figure
feature_importance_fig = px.line(shap_data, x='Date', y=shap_data.columns[:-1],
                                 title='Feature Importance for Predicting Anomalies On Different Dates',
                                 labels={'value': 'SHAP Value', 'Date': 'Date'},
                                 template='plotly')
feature_importance_fig.update_layout(
    title_font_size=28,
    xaxis=dict(tickfont=dict(size=18), title=dict(text="Date", font=dict(size=22))),
    yaxis=dict(tickfont=dict(size=18), title=dict(text="SHAP Value", font=dict(size=22))),
    legend_font_size=22
)

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
        html.Div([dcc.Graph(id='time-series-chart')], className="six columns"),
        html.Div([dcc.Graph(id='correlation-heatmap')], className="six columns"),
        html.Div(id='anomaly-stats', style={'margin-top': '20px', 'text-align': 'center'}),
        html.Div(
            html.Iframe(
                srcDoc=open("Instruments_Image.html").read(),
                style={"height": "600px", "width": "50%", "border": "none"}
            ),
            style={"display": "flex", "justify-content": "center", "align-items": "center"}
        ),
    ], className="row"),
    html.Div([dcc.Graph(id='scaled-time-series-chart')], className="row"),
    html.Div([dcc.Graph(id='anomaly-score-chart')], className="row"),
    html.Div([dcc.Graph(figure=feature_importance_fig, id='feature-importance-chart')])
])

# Callbacks to update graphs
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
    columns_to_scale = [col for col in selected_instruments if col not in ['IBS_R', 'IBS_T', 'IBS_N', 'OBS_R', 'OBS_T', 'OBS_N']]
    scaled_data[columns_to_scale] = scaler.fit_transform(filtered_data[columns_to_scale])

    time_series_fig = go.Figure()
    for instrument in selected_instruments:
        time_series_fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data[instrument], mode='lines+markers', name=instrument))
    time_series_fig.update_layout(
        title="Time Series of Selected Instruments",
        title_font_size=28,
        xaxis=dict(tickfont=dict(size=18), title=dict(text="Date", font=dict(size=22))),
        yaxis=dict(tickfont=dict(size=18), title=dict(text="Instrument Value", font=dict(size=22))),
        legend_font_size=22
    )

    correlation_fig = go.Figure(go.Heatmap(z=scaled_data[selected_instruments].corr(), x=selected_instruments, y=selected_instruments, colorscale='Viridis'))
    correlation_fig.update_layout(
        title="Correlation Heatmap",
        title_font_size=28,
        xaxis=dict(tickfont=dict(size=18), title=dict(text="Instrument", font=dict(size=22))),
        yaxis=dict(tickfont=dict(size=18), title=dict(text="Instrument", font=dict(size=22))),
        legend_font_size=22
    )

    return time_series_fig, correlation_fig, None, None  # Adjust other figures accordingly

if __name__ == "__main__":
    app.run_server(debug=True)
