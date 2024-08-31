import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html, dash_table  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more control over visualizations
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
data_path = "Solar_Orbiter_with_anomalies.csv"  # Path to primary dataset file
data_path2 = "Solar_Orbiter_with_anomalies2.csv"  # Path to secondary dataset file
data_path3 = "Solar_Orbiter_without_units.csv"  # Path to the new dataset for correlation and scaling

solar_data = pd.read_csv(data_path)  # Read primary dataset into DataFrame
solar_data2 = pd.read_csv(data_path2)  # Read secondary dataset into DataFrame
solar_data_no_units = pd.read_csv(data_path3)  # Read new dataset into DataFrame

# Convert the 'Date' columns to datetime format
solar_data['Date'] = pd.to_datetime(solar_data['Date'])
solar_data2['Date'] = pd.to_datetime(solar_data2['Date'])
solar_data_no_units['Date'] = pd.to_datetime(solar_data_no_units['Date'])

# Filter out data from May 5 to May 11
start_exclude = pd.to_datetime('2021-05-05')
end_exclude = pd.to_datetime('2021-05-11')
solar_data = solar_data[~((solar_data['Date'] >= start_exclude) & (solar_data['Date'] <= end_exclude))]
solar_data2 = solar_data2[~((solar_data2['Date'] >= start_exclude) & (solar_data2['Date'] <= end_exclude))]
solar_data_no_units = solar_data_no_units[~((solar_data_no_units['Date'] >= start_exclude) & (solar_data_no_units['Date'] <= end_exclude))]

# Load the SHAP values data
shap_values_path = "shap_values.csv"  # Update with the correct path to your SHAP values CSV
shap_data = pd.read_csv(shap_values_path)

# Create the feature importance figure
feature_importance_fig = px.line(shap_data, x='Date', y=shap_data.columns[:-1],
                                 title='Feature Importance for Predicting Anomalies On Different Dates',
                                 labels={'value': 'SHAP Value', 'Date': 'Date'},
                                 template='plotly')
feature_importance_fig.update_layout(
    title_font_size=28,  # Decrease title font size
    xaxis_title_font_size=22,  # Decrease x-axis title font size
    yaxis_title_font_size=22,  # Decrease y-axis title font size
    legend_font_size=22,  # Decrease legend font size
    xaxis=dict(tickfont=dict(size=18)),  # Set x-axis tick labels size to 18
    yaxis=dict(tickfont=dict(size=18))   # Set y-axis tick labels size to 18
)

# Initialize the Dash app
app = dash.Dash(__name__, title="Solar Orbiter Data Visualization")  # Title of the Dash app which is shown in the browser tab
server = app.server

# Remove the 'Date' and 'anomaly_score' columns from the checklist options
checklist_options = sorted(
    [{'label': col, 'value': col} for col in solar_data.columns if col not in ['Date', 'anomaly_score']],
    key=lambda x: x['label']
)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Solar Orbiter Instrument Data Visualization", style={'text-align': 'center'}),
    
    # Checklist to select instruments
    dcc.Checklist(
        id='instrument-checklist',  # Component ID
        options=checklist_options,  # Options for checklist
        value=[solar_data.columns[1]],  # Default selected value (first instrument)
        inline=True
    ),
    
    # Date range picker
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=solar_data['Date'].min(),  # Minimum date allowed
        max_date_allowed=solar_data['Date'].max(),  # Maximum date allowed
        start_date=solar_data['Date'].min(),  # Default start date
        end_date=solar_data['Date'].max()  # Default end date
    ),
    
    # Three rows, each containing graphs
    html.Div([
        html.Div([dcc.Graph(id='time-series-chart')], className="six columns"),  # Time Series Chart

        html.Div([dcc.Graph(id='correlation-heatmap')], className="six columns"),
        html.Div(id='anomaly-stats', style={'margin-top': '20px', 'text-align': 'center'}),  # Anomaly Stats
        
        html.Div(
            html.Iframe(
                srcDoc=open("Instruments_Image.html").read(),
                style={"height": "600px", "width": "50%", "border": "none"}
            ),
            style={"display": "flex", "justify-content": "center", "align-items": "center"}
        ),  # Instruments Image
    ], className="row"),
        
    html.Div([
        html.Div([dcc.Graph(id='scaled-time-series-chart')], className="six columns"),  # Scaled Time Series Chart
    ], className="row"),
    
    html.Div([
        html.Div([dcc.Graph(id='anomaly-score-chart')], className="six columns"),
    ], className="row"),

    # Add the feature importance graph at the bottom
    html.Div([
        dcc.Graph(figure=feature_importance_fig, id='feature-importance-chart')
    ])
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
    # Convert start_date and end_date to datetime if necessary
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the new dataset for correlation heatmap and scaled plots
    filtered_data_no_units = solar_data_no_units[(solar_data_no_units['Date'] >= start_date) & (solar_data_no_units['Date'] <= end_date)]
    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]
    filtered_data2 = solar_data2[(solar_data2['Date'] >= start_date) & (solar_data2['Date'] <= end_date)]
    
    # Normalize the data from the new dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data_no_units = filtered_data_no_units.copy()

    # Ensure that only selected instruments are scaled
    if selected_instruments:
        scaled_data_no_units[selected_instruments] = scaler.fit_transform(filtered_data_no_units[selected_instruments])

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
        xaxis_title='Date',
        yaxis_title='Value'
    )
    
    # Correlation Heatmap
    if len(selected_instruments) > 1:
        correlation_matrix = filtered_data_no_units[selected_instruments].corr()
        correlation_fig = go.Figure(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='Viridis'
            )
        )
        correlation_fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title='Instruments',
            yaxis_title='Instruments'
        )
    else:
        correlation_fig = go.Figure()  # Empty figure if not enough instruments selected

    # Scaled Time Series Chart
    scaled_time_series_fig = go.Figure()
    for instrument in selected_instruments:
        scaled_time_series_fig.add_trace(
            go.Scatter(
                x=scaled_data_no_units['Date'],
                y=scaled_data_no_units[instrument],
                mode='lines+markers',
                name=instrument
            )
        )
    scaled_time_series_fig.update_layout(
        title="Scaled Time Series Plot between -1 and 1",
        xaxis_title='Date',
        yaxis_title='Scaled Value'
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
            size=5,
            line=dict(color='DarkSlateGrey', width=2)
        )
    ))
    anomaly_score_fig.update_layout(
        title="Anomaly Scores Over Time",
        xaxis_title='Date',
        yaxis_title='Anomaly Score'
    )

    return time_series_fig, correlation_fig, scaled_time_series_fig, anomaly_score_fig



"""References:
1. https://dash.plotly.com/ - Dash Documentation
2. https://dash.plotly.com/layout - Dash Layout (HTML Components)
3. https://dash.plotly.com/dash-core-components - Dash Core Components (DatePickerRange, Checklist)
4. https://dash.plotly.com/dash-html-components - Dash HTML Components (Div, H1, Iframe)
5. https://plotly.com/python/plotly-express/ - Plotly Express (px.line, px.scatter, px.bar)
6. https://plotly.com/python/graph-objects/ - Plotly Graph Objects (go.Scatter, go.Heatmap, go.Figure)
7. https://www.coursera.org/projects/interactive-dashboards-plotly-dash?tab=guided-projects - Coursera Project
"""

if __name__ == "__main__":
    app.run_server(debug=True)  # Start the Dash server in debug mode
