#### basic packages
import warnings
warnings.filterwarnings("ignore")

#### computation packages
import numpy as np
import pandas as pd

#### plotly&dash packages
import dash
import plotly.graph_objects as go
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
# from dash import dcc
# from dash import html
from dash.dependencies import Input, Output

from prophet import Prophet

#############################################################################
#### load data ##############################################################
#############################################################################
path = "/Users/andy/Desktop/py_project/kaggle/G-Research/data"

details = pd.read_csv(path + "/asset_details.csv")
train = pd.read_csv(path + "/train.csv")


# modify the label
asset_name = ['Binance Coin', 'Bitcoin', 'Bitcoin Cash', 'Cardano', 'Dogecoin', 'EOS.IO', 'Ethereum', 
              'Ethereum Classic', 'IOTA', 'Litecoin', 'Maker', 'Monero', 'Stellar', 'TRON']

for index in range(0,14):
    train['Asset_ID'].replace({index : asset_name[index]}, inplace=True)

# timestamp as index
train['timestamp'] = pd.to_datetime(train['timestamp'], unit='s')
train.set_index('timestamp', inplace = True)


# Making separate trainframes

bitcoin_cash = train[train['Asset_ID'] == 'Bitcoin Cash']
binance_coin = train[train['Asset_ID'] == 'Binance Coin']
bitcoin = train[train['Asset_ID'] == 'Bitcoin']
eos_io = train[train['Asset_ID'] == 'EOS.IO']
ethereum_classic = train[train['Asset_ID'] == 'Ethereum Classic']
ethereum = train[train['Asset_ID'] == 'Ethereum']
litecoin = train[train['Asset_ID'] == 'Litecoin']
monero = train[train['Asset_ID'] == 'Monero']
tron = train[train['Asset_ID'] == 'TRON']
stellar = train[train['Asset_ID'] == 'Stellar']
cardano = train[train['Asset_ID'] == 'Cardano']
iota = train[train['Asset_ID'] == 'IOTA']
maker = train[train['Asset_ID'] == 'Maker']
dogecoin = train[train['Asset_ID'] == 'Dogecoin']

day = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}

bitcoin_cash = bitcoin_cash.resample('D').apply(day)
binance_coin = binance_coin.resample('D').apply(day)
bitcoin = bitcoin.resample('D').apply(day)
eos_io = eos_io.resample('D').apply(day)
ethereum_classic = ethereum_classic.resample('D').apply(day)
ethereum = ethereum.resample('D').apply(day)
litecoin = litecoin.resample('D').apply(day)
monero = monero.resample('D').apply(day)
tron = tron.resample('D').apply(day)
stellar = stellar.resample('D').apply(day)
cardano = cardano.resample('D').apply(day)
iota = iota.resample('D').apply(day)
maker = maker.resample('D').apply(day)
dogecoin = dogecoin.resample('D').apply(day)

df_list = [bitcoin_cash, binance_coin, bitcoin, eos_io, ethereum_classic, ethereum,
           litecoin, monero, tron, stellar, cardano, iota, maker, dogecoin]

# reference: https://www.kaggle.com/toomuchsauce/g-crypto-interactive-dashboard-indicators


##############################################################################################
#### Weight ##################################################################################
##############################################################################################

details_plot = details.sort_values("Weight")
fig_index = px.bar(details_plot, x="Asset_Name" , y="Weight", color="Weight", title="Popular Cryptocurrency Weight Distribution")


#--- Treemap ----------------------------------------------------------------- #

count_mean = train[['Asset_ID', 'Count']].groupby('Asset_ID').mean()
count_mean.reset_index(level=0,inplace = True)
count_mean = count_mean.rename(columns = {'Asset_ID': 'Asset_Name'})
details_plot = details_plot.merge(count_mean, on='Asset_Name')
details_plot

# treemap
fig_treemap = px.treemap(details_plot, 
        names = "Asset_Name", parents=[""] * len(details_plot["Asset_Name"]), values="Weight", \
                         color="Count", color_continuous_scale = 'RdBu', title="Popular Cryptocurrency Tree map"
    )



#---- Candel plot ------------------------------------------------------------ #
names = train.Asset_ID.unique()

date_buttons = [
{'count':  4, 'step': "year", 'stepmode': "todate", 'label': "Year"},
{'count':  6, 'step': "month", 'stepmode': "todate", 'label': "Month"},
{'count': 60, 'step': "day", 'stepmode': "todate", 'label': "Week"},
{'count': 20, 'step': "day", 'stepmode': "todate", 'label': "Day"},
               ]

buttons = []
i = 0
vis = [False] * 14

for df in df_list:
    vis[i] = True
    buttons.append({ 'label' : names[i],
                     'method' : 'update',
                     'args'   : [{'visible' : vis},
                                 {'title'  : names[i] + ' Chart'}] })
    i+=1
    vis = [False] * 14

fig = go.Figure()

for df in df_list:
    fig.add_trace(go.Candlestick(x     = df.index,
                                 open  = df['Open'],
                                 high  = df['High'],
                                 low   = df['Low'],
                                 close = df['Close'],
                                 increasing_line_color= '#3D9970',
                                 decreasing_line_color= '#FF4136'))

fig.update_xaxes(
        # tickfont = dict(size=15, family = 'monospace', color='#B8B8B8'),
        tickmode = 'array',
        ticklen = 6,
        showline = False,
        showgrid = True,
        # gridcolor = '#595959',
        ticks = 'outside')

fig.update_yaxes(
        # tickfont = dict(size=15, family = 'monospace', color='#B8B8B8'),
        tickmode = 'array',
        showline = False,
        ticksuffix = '$',
        showgrid = True,
        # gridcolor = '#595959',
        ticks = 'outside')    
    
    
fig.update_layout(height = 800,
                  font_family   = 'monospace',
                  xaxis         = dict(rangeselector = dict(buttons = date_buttons)),
                  updatemenus   = [dict(type = 'dropdown',
                                        x = 1,
                                        y = 1.15,
                                        showactive = True,
                                        active = 2,
                                        buttons = buttons)],
                  title         = dict(text = 'Cryptocurrency Candlestick & Volume<br>Dashboard',
                                       font = dict(color = '#B8B8B8'), 
                                       x = 0.525),
                  annotations   = [
                                  dict( text = "<b>Choose<br>Crypto<b> : ",
                                        font = dict(size = 12),
                                        showarrow=False,
                                        x = 0.96, y = 1.25,
                                        xref = 'paper', yref = "paper",
                                        align = "left")])
    
    
for i in range(0,14):
    fig.data[i].visible = False
fig.data[2].visible = True


#---- Volume ------------------------------------------------ #


v = {'bitcoin_cash' : bitcoin_cash['Volume'],
     'binance_coin' : binance_coin['Volume'],
     'bitcoin' : bitcoin['Volume'],
     'eos_io' : eos_io['Volume'],
     'ethereum_classic' : ethereum_classic['Volume'],
     'ethereum' : ethereum['Volume'],
     'litecoin' : litecoin['Volume'],
     'monero' : monero['Volume'],
     'tron' : tron['Volume'],
     'stellar' : stellar['Volume'],
     'cardano' : cardano['Volume'],
     'iota' : iota['Volume'],
     'maker' : maker['Volume'],
     'dogecoin' : dogecoin['Volume']}


df_volume = pd.DataFrame(data = v)
fig_volume = px.line(df_volume, x = df_volume.index, y = df_volume.columns,
              title='Volume')

fig_volume.update_layout(height = 650,
                  font_family   = 'monospace',
                  title         = dict(text = 'Cryptocurrency Volume<br>Dashboard',
                                       font = dict(color = '#B8B8B8'), 
                                       x = 0.525))

fig_volume.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    ticklabelmode="period")


###########################################################################
##### Time series predict #################################################
###########################################################################

def time_series(df):

    model = Prophet(daily_seasonality=True)

    df_prophet = df.drop(['Open', 'High', 'Low','Volume'], axis=1)
    df_prophet.reset_index(level=0, inplace=True)
    df_prophet.rename(columns={'Close': 'y', 'timestamp': 'ds'}, inplace=True)

    model.fit(df_prophet)

    future_prices = model.make_future_dataframe(periods=365)
    forecast = model.predict(future_prices)
    df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return df_prophet, df_forecast

df_prophet, df_forecast = time_series(bitcoin)
test = pd.read_csv(path + "/test.csv")
test['ds'] = pd.to_datetime(test['Date'])
df_real = test.merge(df_forecast[1359:1451], on='ds', how='left').drop('Date', axis = 1).dropna()

trace_open = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat"],
    mode = 'lines',
    name="Forecast"
)

trace_high = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat_upper"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Higher uncertainty interval"
)

trace_low = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat_lower"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Lower uncertainty interval"
)

trace_close = go.Scatter(
    x = df_prophet["ds"],
    y = df_prophet["y"],
    name="Data values"
)

trace_real_close = go.Scatter(
    x = test["ds"],
    y = test["Close*"],
    name="Real Close values"
)
trace_real_open = go.Scatter(
    x = test["ds"],
    y = test["Open"],
    name="Real Open values"
)

data = [trace_open,trace_high,trace_low,trace_close, trace_real_close, trace_real_open]

layout = go.Layout(title="Bitcoin Price Forecast", xaxis_rangeslider_visible=True)

fig_predict = go.Figure(data=data,layout=layout)

fig_predict.update_layout(height = 800,
                  font_family   = 'monospace',
                  title         = dict(text = 'Cryptocurrency Predict',
                                       font = dict(color = '#B8B8B8'), 
                                       x = 0.525))
                                       
###########################################################################
##### Dash ################################################################
###########################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
        
])

about_page = html.Div([
    html.Div([
        html.H1("Cryptocurrency Analytics")
    ], className="Topbar"),

    html.Div([
        html.Ul([
            html.Li(dcc.Link('About', href='/about', className='lin')),
            html.Li(dcc.Link('Candlestick & Volume & Volume', href='/main', className='lin')),
            html.Li(dcc.Link('Distribution', href='/two', className='lin')),
            html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
            ]),
    ], className="Sidebar"),

    html.Div([
        html.H3("Background:"),
        html.Br(),
        html.P("Bitcoin reached an all-time high of $68,000 last November,"),
        html.Br(),
        html.P("but it has been volatile and has slipped to a low of $46,000 several times in the past few months."),
        html.Br(),
        html.P("Still, many experts say that Bitcoin is heading towards the $100,000 mark, but when?"),
        html.Br(),
        html.H3("Aim:"),
        html.Br(),
        html.P("To analyze the Cryptocurrency Daily Price and visualize the weight of each asset between 2018 and 2021,"),
        html.Br(),
        html.P("and try to predict the price in such volatile circumstances."),
        html.Br(),
        html.H3("Data:"),
        html.Br(),
        html.P("G-Research Competition 2018-2020 Cryptocurrency"),
    ], className="About")
])


# main_graph_page = html.Div([
#     html.Div([
#         html.H1("Cryptocurrency Analytics")
#     ], className="Topbar"),

#     html.Div([
#         html.Ul([
#             html.Li(dcc.Link('About', href='/about', className='lin')),
#             html.Li(dcc.Link('Candlestick & Volume', href='/main', className='lin')),
#             html.Li(dcc.Link('Distribution', href='/two', className='lin')),
#             html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
#             ]),
#     ], className="Sidebar"),

#     html.Div([
#         dcc.Graph(figure= fig, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block'}, className="Plot1"),
#         dcc.Graph(figure = fig_volume, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block'}, className="Plot3")
#     ], className="Main")
# ])

main_graph_page = html.Div([
    html.Div([
        html.H1("Cryptocurrency Analytics")
    ], className="Topbar"),

    html.Div([
        html.Ul([
            html.Li(dcc.Link('About', href='/about', className='lin')),
            html.Li(dcc.Link('Candlestick & Volume', href='/main', className='lin')),
            html.Li(dcc.Link('Distribution', href='/two', className='lin')),
            html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
            ]),
    ], className="Sidebar"),

    html.Div([
        dcc.Graph(figure= fig, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block'}, className="Plot1"),
        dcc.Graph(figure = fig_volume, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block'}, className="Plot3")
    ], className="Main"),
])



two_graph_page = html.Div([
    html.Div([
        html.H1("Cryptocurrency Analytics")
    ], className="Topbar"),

    html.Div([
        html.Ul([
            html.Li(dcc.Link('About', href='/about', className='lin')),
            html.Li(dcc.Link('Candlestick & Volume', href='/main', className='lin')),
            html.Li(dcc.Link('Distribution', href='/two', className='lin')),
            html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
            ]),
    ], className="Sidebar"),

    html.Div([
        dcc.Graph(id="graph1", figure= fig_index, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block', 'padding': '0 5px 0 5px'}, className="Plot2"),
        dcc.Graph(id="graph2", figure= fig_treemap, style={'height': '80vh', 'width': '40vw', 'display': 'inline-block', 'padding': '0 5px 0 5px'}, className="Plot2")
    ], className="Main")
])


volume_predict_page = html.Div([
    html.Div([
        html.H1("Cryptocurrency Analytics")
    ], className="Topbar"),

    html.Div([
        html.Ul([
            html.Li(dcc.Link('About', href='/about', className='lin')),
            html.Li(dcc.Link('Candlestick & Volume', href='/main', className='lin')),
            html.Li(dcc.Link('Distribution', href='/two', className='lin')),
            html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
            ]),
    ], className="Sidebar"),

    html.Div([
        dcc.Graph(figure = fig_predict, style={'height': '80vh', 'width': '80vw', 'display': 'inline-block'}, className="Plot3")
    ], className="Main")
])


# conclusion_page = html.Div([
#     html.Div([
#         html.H1("Cryptocurrency Analytics")
#     ], className="Topbar"),

#     html.Div([
#         html.Ul([
#             html.Li(dcc.Link('About', href='/about', className='lin')),
#             html.Li(dcc.Link('Candlestick & Volume', href='/main', className='lin')),
#             html.Li(dcc.Link('Distribution', href='/two', className='lin')),
#             html.Li(dcc.Link('Predict', href='/volume-predict', className='lin')),
#             ]),
#     ], className="Sidebar"),

#     html.Div([
#         html.P(children="Analyze the Cryptocurrency Daily Price"
#                             " and visualize the weight of each asset"
#                             " between 2018 and 2021")
#     ], className="Main")
# ])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/main':
        return main_graph_page
    elif pathname == '/two':
        return two_graph_page
    elif pathname == '/volume-predict':
        return volume_predict_page
    # elif pathname == '/conclusion':
    #     return conclusion_page
    else:
        return about_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, host = '127.0.0.1')  # Turn off reloader if inside Jupyter