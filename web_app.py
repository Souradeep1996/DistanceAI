import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import pandas as pd
import datetime

df = pd.read_csv("Violations_DIP.csv")

app.layout = html.Div(children=[
    html.H1(children='Distance AI Dashboard', style={'text-align': 'center'}),
    dcc.Graph(
        id='violations-graph', animate=True   
    ),
    dcc.Interval(
            id='graph-update',
            interval=5*1000),
    html.Div(children='''Powered by TML Digital Core Team''', style={'text-align': 'center'}),
])

@app.callback(Output('violations-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph(input_data):
    df = pd.read_csv("Violations_DIP.csv")
    todays_date = datetime.datetime.now().date()
    index = pd.date_range(start=todays_date-datetime.timedelta(6), end= todays_date,  freq='D')
    return {
            'data': [
                {'x': index, 'y': df['DIP Gate'], 'type': 'bar', 'name': 'DIP Gate'},
                {'x': index, 'y': df['Main Gate'], 'type': 'bar', 'name': 'Main Gate'},
            ],
            'layout': {
                'title': 'Social Distancing Violations',
            }
        }

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port = 8080)
