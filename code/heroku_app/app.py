import pathlib
import os
import pandas as pd
import datetime as dt
import getpass
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash.dependencies import Input, Output, State
import base64
from wordcloud_fcn import *
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# how to get df_rated1.pkl from urls_2020.csv, run this part in local to get df_rated1.pkl
# df_rated = pd.read_csv('urls_2020.csv')
# del df_rated['Unnamed: 0']
# del df_rated['Unnamed: 0.1']
# df_rated = df_rated[df_rated['date']>='2020-05-01']
# df_rated.to_pickle('df_rated1.pkl')

# heatmap data
df_rated = pd.read_pickle('df_rated1.pkl')
thresh= pd.read_csv('thresh.csv')
outl_d = sorted(list(set(df_rated['date'])))
ind_lst = list(set(df_rated['GICS Sector']))

df_diff = df_rated[['MA_7day_1', 'MA_7day_2', 'MA_7day_3',
       'MA_7day_4', 'MA_7day_5', 'MA_7day_6', 'MA_7day_7', 'MA_7day_8',
       'MA_7day_9', 'MA_7day_10', 'MA_7day_11', 'MA_7day_12', 'MA_7day_13',
       'MA_7day_14', 'MA_7day_15', 'MA_7day_16', 'MA_7day_17', 'MA_7day_Mean']].diff()
columns_lst = ['diff_'+str(i) for i in range(1,18)]
columns_lst.append('diff_Mean')
df_diff.columns = columns_lst

df_merge = pd.concat([df_rated,df_diff],axis=1)
first = min(df_merge['date'])
indexer = df_merge[df_merge['date']== first].index
df_merge.loc[indexer, columns_lst]= float("NAN")

del df_diff

thecolumns = list(df_rated.columns)
del df_rated


def level(thresh, MA, DIFF,count_, std_, sdg, freq):
    if freq == "day":
        ind=0
    elif freq == "week":
        ind=1
    else:
        ind=2

    scale=thresh['level'][ind]
    count=thresh['count'][ind]
    std=thresh['std'][ind]
    diff=thresh['change'][ind]
    count2 = thresh['SDG_'+str(sdg)+'_count'][ind]
    std2 = thresh['SDG_'+str(sdg)+'_std'][ind]

    if (abs(MA) < scale) or (abs(DIFF) < diff):
        return 0
    if MA>0:
        if count_ >= count2 and std_ <= std2:
            return 2
        else:
            return 1
    elif MA < 0:
        if count_ >= count2 and std_ <= std2:
            return -2
        else:
            return -1
    else: return 0


def heatmapf(day, industry, thresh, freq='day'):
    col = ["COMPANY"]
    for sdg in range(1,18):
        col.append('MA_7day_'+str(sdg))
    df = df_merge[(df_merge['GICS Sector']==industry) & (df_merge['date']==day)].reset_index(drop=True)
    score =  df[col]
    # generate a color label table
    color = pd.DataFrame(columns=col)
    color['COMPANY']=score['COMPANY']
    """
    Outlier parameter
    """
    for sdg in range(1,18):
        for i in range(len(df)):
            MA = df['MA_7day_'+str(sdg)][i]
            DIFF = df['diff_'+str(sdg)][i]
            count_ = df['SDG_' + str(sdg) + '_count'][i]
            std_ = df['SDG_' + str(sdg) + '_std'][i]
            color['MA_7day_'+str(sdg)][i] = level(thresh, MA, DIFF, count_, std_, sdg, freq)
    # replace nan with '-'
    for c in col[1:]:
        for i in range(len(score)):
            if abs(score[c][i]) <= 100:
                score[c][i] = round(score[c][i],2)
            else: score[c][i] = '-'
    del df
    return color, score


# wordcloud data
wc_date = sorted(list(set(df_merge[thecolumns]['date'])))
SDGs =[i for i in range(1,18)]

# line graph data
def return_line_graph_data(df,company, date):
    dateList = []
    for i in range(-10,11):
        dateList.append(str(datetime.datetime.strptime(date,'%Y-%m-%d') + datetime.timedelta(days=i))[:10])
    col = col = [dateList[x]+'(T'+str(x-10)+')' for x in range(0,21)]
    col = ['sdg' ]+col
    result = pd.DataFrame(columns=col)
    df = df[(df['COMPANY']==company)]

    for sdg in range(1,18):
        row = []
        for element in dateList:
            try:
                sample = df[df['date']==element]
                row.append(list(sample['SDG_'+str(sdg)])[0])   # modified row
            except:
                row.append(None)
        row = ['SDG'+str(sdg)]+row
        new_row = {}
        for i in range(len(row)):
            new_row[col[i]] = row[i]
        result = result.append(new_row,ignore_index=True)
    return result



companies = df_merge[thecolumns]['COMPANY'].unique().tolist()
daterange = df_merge[thecolumns]['date'].unique().tolist()
min_date = dt.datetime.strptime(daterange[0],'%Y-%m-%d').date()
max_date = dt.datetime.strptime(daterange[-1],'%Y-%m-%d').date()

#%%
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
app.config['suppress_callback_exceptions'] = True
server = app.server
app.layout = html.Div([
    html.H1('Word Cloud', style={'textAlign': 'Center', 'color': '#CD5C5C', 'margin-top': 20}),
    html.Img(
        src='https://images.squarespace-cdn.com/content/5c036cd54eddec1d4ff1c1eb/1557908564936-YSBRPFCGYV2CE43OHI7F/GlobalAI_logo.jpg?content-type=image%2Fpng',
        style={
            'height': '15%',
            'width': '15%',
            'float': 'right',
            'position': 'relative',
            'margin-top': 20,
            'margin-left': 5,
            'margin-right': 0}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Tabs(id="tabs", value='tab_1', children=[
            dcc.Tab(label='Outlier Line Graph', value='tab_1',
                    style={"height": 60, 'font-size': '15px'},
                    selected_style={"height": 60, 'font-size': '15px'}),
            dcc.Tab(label='Outlier Heatmap Graph', value='tab_2',
                    style={"height": 60, 'font-size': '15px'},
                    selected_style={"height": 60, 'font-size': '15px'}),
            dcc.Tab(label='Word Cloud', value='tab_3',
                    style={"height": 60, 'font-size': '15px'},
                    selected_style={"height": 60, 'font-size': '15px'})],
                 colors={"border": "gray",
                         "primary": "gray",
                         "background": "lightgoldenrodyellow"},
                 style={'textAlign': 'Center', 'color': 'black'})],
             style={"height": "40px"}),
    html.Br(),
    html.Br(),
    html.Div(style=dict(clear="both")),
    html.Div(id='tabs_content')], style={"padding": "40px", "background": "black"})

@app.callback(Output('tabs_content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab_1':
        return tab1_layout
    elif tab == 'tab_2':
        return tab2_layout
    elif tab == 'tab_3':
        return tab3_layout



# ---------------------   tab_1 ----------------------------
tab1_layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3('Companies:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label1', value='american airlines group', options=[{'label': i, 'value': i} for i in companies],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'})),
            dbc.Col(
                html.Div([
                    html.H3('Date:', style={'paddingRight': '35px'}),
                    dcc.DatePickerSingle(
                        id='my-date-picker-single',
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        date=dt.date(2020, 5, 12)
                        )],
                    style={'verticalAlign': 'top', 'display': 'inline-block'}))]),
        html.Br(),
        dbc.Button("Generate Graphs", color="primary",
                   block=True, id="button1", n_clicks=1, className="mb-3"),
        html.Br(),
        dbc.Row([dbc.Col(dcc.Graph('Time_Series'))])])])


@app.callback(
    Output("Time_Series", "figure"),
    [Input("button1", "n_clicks")],
    [State("label1", "value"),
     State("my-date-picker-single", "date")])
def updatefunction1(n_clicks, label, date):
    date = date[0:10]
    res = return_line_graph_data(df_merge[thecolumns],company = label, date=date)
    sdg_data = res.iloc[:,1:]

    length = len(sdg_data)
    trace=[]
    for i in range(length):
        trace.append(
                go.Scatter(
                    x=sdg_data.columns.tolist(),
                    y=sdg_data.loc[sdg_data.index[i]].tolist(),
                    mode="lines",
                    name=res.sdg.tolist()[i]))
    timeseries = go.Figure(
                        data=trace,
                        layout={'title': {'text': f"{label} SDG line plot<br>{date}",
                                          'xanchor': 'center', 'x': 0.5},
                                'xaxis': {'title': 'Date'},
                                'yaxis': {'title': 'SDG'},
                                'paper_bgcolor': "Black",
                                'plot_bgcolor': 'Black',
                                'font': {'color': '#7FDBFF'}})

    return timeseries

# ---------------------   tab_2 ----------------------------
tab2_layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3('Outlier Date:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label-outld', value= outl_d[0], options=[{'label': i, 'value': i} for i in outl_d],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'})),
            dbc.Col(
                html.Div([
                    html.H3('Industry:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label-ind', value=ind_lst[0], options=[{'label': i, 'value': i} for i in ind_lst],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'}))
                         ]),
        html.Br(),
        dbc.Button("Generate Graphs", color="primary",
                   block=True, id="button2", n_clicks=1, className="mb-3"),
        html.Br(),
        html.H1('Outlier Heatmap Graph', style={'textAlign': 'Center', 'color':'lightskyblue','font-size': '20px'}),
        html.Br(),
        dbc.Row([dbc.Col(dcc.Graph('Heatmap',style={'margin-top':'20px','height':2000}))])])])


@app.callback(
    Output("Heatmap", "figure"),
    [Input("button2", "n_clicks")],
    [State("label-outld", "value"),
    State("label-ind","value")])
def updatefunction2(n_clicks, label, label1):

    color, score = heatmapf(day=label,industry=label1,thresh=thresh)

    x = list(color.columns[1:])
    y = list(color['COMPANY'])
    z = color.iloc[:,1:].values
    z_text = score.iloc[:,1:].values
    white_p = 0 - np.min(z) / (np.max(z) - np.min(z))
    colorscale = [[0, 'red'], [white_p, 'white'], [1, 'skyblue']]
    font_colors = ['black', 'black']
    heatmap = ff.create_annotated_heatmap(z, x=x, y=y,annotation_text=z_text,colorscale=colorscale, font_colors=font_colors)

    heatmap.update_layout(
                       xaxis={'title':'SDG'},
                       yaxis={'title': 'Company'},
                       paper_bgcolor="Black",
                       plot_bgcolor="Black",
                       font={"color": "#7FDBFF"})

    return heatmap

# ---------------------   tab_3 ----------------------------
tab3_layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3('Companies:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label-comp', value='apple inc', options=[{'label': i, 'value': i} for i in companies],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'})),
            dbc.Col(
                html.Div([
                    html.H3('SDG:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label-sdg', value=1, options=[{'label': i, 'value': i} for i in SDGs],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'})),
            dbc.Col(
                html.Div([
                    html.H3('Outlier Date:', style={'paddingRight': '35px'}),
                    dcc.Dropdown(id='label-date', value= wc_date[11], options=[{'label': i, 'value': i} for i in wc_date],
                                 style={'backgroundColor': 'white', 'color': 'black'})],
                         style={'verticalAlign': 'top', 'display': 'inline-block'}))]),
        html.Br(),
        dbc.Button("Generate Graphs", color="primary",
                   block=True, id="button3", n_clicks=1, className="mb-3"),
        html.Br(),
        dbc.Row([
        dbc.Col(dcc.Graph('table-neg-score',style={'height':500})),

        dbc.Col(
        html.Div([html.H3('Word Cloud (negative keywords)',style={'textAlign': 'Center', 'color':'lightskyblue','font-size': '20px'}),
        html.Br(),
        dbc.Row([html.Div([html.Img(id='word_cloud', style={'height': '90%', 'width': '90%'})])], style={'textAlign': 'center'})])
        ),

        dbc.Col(dcc.Graph('table-urls',style={'height':500}))
        ])
        ])])


@app.callback(
    [Output('table-neg-score', 'figure'),
    Output("word_cloud","src"),
    Output('table-urls', 'figure')],
    [Input("button3", "n_clicks")],
    [State("label-comp", "value"),
    State("label-sdg", "value"),
    State("label-date", "value")])
def updatefunction3(n_clicks, comp, sdg, date):

    # data for negative scores table
    negscore = return_neg_score(df_merge[thecolumns],company=comp,sdg=sdg,date=date)
    neg_score_fig = go.Figure(data=[go.Table(
    header=dict(values=list(negscore.columns),
                font=dict(size=10, color="#fff"),
                align="left",
                fill_color='#222'),
    cells=dict(values=[negscore.date, negscore.score],
               fill_color='grey',
               font=dict(color='white'),
               align='left'))])
    neg_score_fig.update_layout(title=dict(text="Sentiment Scores (negtive outliers)",
    xanchor='center',x=0.5),paper_bgcolor="Black",plot_bgcolor="Black",font={"color": "#7FDBFF"})
    del negscore


    # data for word cloud graph
    return_wordcloud(df_merge[thecolumns],sdg = sdg,company=comp,date=date)
    image_filename1 = 'word_cloud_image.png'
    encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())
    wc_image = 'data:image/png;base64,{}'.format(encoded_image1.decode())

    # data for urls
    df_url = pd.DataFrame(columns=['urls'])
    df_url['urls']=return_url(df_merge[thecolumns],sdg=sdg,company=comp,date=date)
    url_fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_url.columns),
                font=dict(size=10, color="#fff"),
                align="left",
                fill_color='#222'),
    cells=dict(values=[df_url.urls],
               fill_color='grey',
               font=dict(color='white'),
               align='left'))])
    url_fig.update_layout(title=dict(text="Sample News (during event period)",
    xanchor='center',x=0.5),paper_bgcolor="Black",plot_bgcolor="Black",font={"color": "#7FDBFF"})
    del df_url

    return neg_score_fig, wc_image, url_fig

if __name__ == '__main__':
    app.run_server(debug=False)