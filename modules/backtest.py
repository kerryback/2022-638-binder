import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pandas_datareader import DataReader as pdr

def backtest(data, features, target, pipe, numstocks):
    df = None
    dates = data.date.unique()
    train_dates = ["2005-01", "2010-01", "2015-01", "2020-01"]
    end_months = ["2009-12", "2014-12", "2019-12", "2022-03"]
    for start, end in zip(train_dates, end_months):
        past = data[data.date<start]
        X = past[features]
        y = past[target]
        pipe.fit(X, y)
        predict_dates = [d for d in dates if d>=start and d<=end]
        for d in predict_dates:
            present = data[data.date==d]
            X = present[features]
            out = pd.DataFrame(dtype=float, columns=["date", "ticker", "predict", "ret"])
            out["ticker"] = present.ticker
            out["predict"] = pipe.predict(X)
            out["date"] = d
            out["ret"] = present.ret 
            df = pd.concat((df, out))
    
    numstocks = numstocks if isinstance(numstocks, list) else [numstocks]
    lst = []
    for num in numstocks:
        df["rnk"] = df.groupby("date").predict.rank(method="first", ascending=False)
        best = df[df.rnk<=num]
        df["rnk"] = df.groupby("date").predict.rank(method="first")
        worst = df[df.rnk<=num]

        best_rets = best.groupby("date").ret.mean()
        worst_rets = worst.groupby("date").ret.mean()
        rets = pd.concat((best_rets, worst_rets), axis=1)
        rets.columns = ["best", "worst"]
        lst.append(rets)
    return rets if len(rets)>1 else rets[0]

def cumplot(rets):
    traces = []
    for ret in ["best", "market", "worst"]:
        trace = go.Scatter(
            x=rets.date,
            y=(1+rets[ret]).cumprod(),
            mode="lines",
            name=ret,
            hovertemplate=ret + " = $%{y:.2f}<extra></extra>"
        )
        traces.append(trace)
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickprefix="$",
        hovermode="x unified",
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01
        )
    )
    return fig

def mvplot(df):
    rets = df.copy().set_index("date")[["best", "market", "worst"]]
    r1, r2, r3 = rets.columns.to_list()
    mns = 12*rets.mean()
    sds = np.sqrt(12)*rets.std()
    cov = 12*rets.cov()
    rf = pdr("DGS1MO", "fred").iloc[-1].item() / 100
    rprem = mns - rf
    w = np.linalg.solve(cov, rprem)
    w = w / np.sum(w)
    mn = w @ mns
    sd = np.sqrt(w @ cov @ w)
    mxsd = np.max(sds)

    w1 = np.linalg.solve(cov, np.ones(3))
    w1 = w1 / np.sum(w1)
    mn1 = w1 @ mns
    w2 = np.linalg.solve(cov, mns)
    w2 = w2 / np.sum(w2)
    mn2 = w2 @ mns
    def port(m):
        a = (m-mn2) / (mn1-mn2)
        return a*w1 + (1-a)*w2
    
    traces = []
    for ret in ["best", "market", "worst"]:
        trace = go.Scatter(
            x=[sds[ret]],
            y=[mns[ret]],
            mode="markers",
            marker=dict(size=10),
            hovertemplate=ret+"<br>mean=%{y:.1%}<br>stdev=%{x:.1%}<extra></extra>",
            name=ret,
        )
        traces.append(trace)

    cd = np.empty(shape=(1, 3, 1), dtype=float)
    cd[:, 0] = np.array(w[0])
    cd[:, 1] = np.array(w[1])
    cd[:, 2] = np.array(w[2])
    string = "Tangency portfolio:<br>"
    string += "best: %{customdata[0]:.1%}<br>"
    string += "market: %{customdata[1]:.1%}<br>"
    string += "worst: %{customdata[2]:.1%}<br>"
    string += "<extra></extra>"
    trace = go.Scatter(
        x=[sd],
        y=[mn],
        mode="markers",
        marker=dict(size=10),
        customdata=cd,
        hovertemplate=string,
        name="tangency"
    )
    traces.append(trace)

    x = np.linspace(0, mxsd, 51)
    y = rf+x*(mn-rf)/sd
    trace = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        hovertemplate=f"Sharpe ratio = {(mn-rf)/sd:0.1%}<extra></extra>",
        showlegend=False,
    )
    traces.append(trace)

    maxmn = np.max(y)
    ms = np.linspace(np.min(mns), maxmn, 51)
    ps = [port(m) for m in ms]
    ss = [np.sqrt(p@cov@p) for p in ps]
    cd = np.empty(shape=(len(ps), 3, 1), dtype=float)
    for i in range(3):
        cd[:, i] = np.array([w[i] for w in ps]).reshape(-1, 1)
    string = "best = %{customdata[0]:.1%}<br>" 
    string += "market = %{customdata[1]:.1%}<br>"
    string += "worst = %{customdata[2]:.1%}<br>"
    string += "<extra></extra>"
    trace = go.Scatter(
        x=ss,
        y=ms,
        mode="lines",
        customdata=cd,
        hovertemplate=string,
        showlegend=False,
    )
    traces.append(trace)

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        yaxis_title="Annualized Mean",
        xaxis_title="Annualized Standard Deviation",
        xaxis_rangemode="tozero",
        yaxis_rangemode="tozero",
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01
        )
    )
    return fig

import statsmodels.formula.api as smf

def regress(rets, y):
    x = 100*12*(rets.market - rets.rf)
    y = 100*12*(rets[y] - rets.rf)
    df = pd.concat((x, y), axis=1)
    result = smf.ols("y~x", df).fit()
    table = result.summary2().tables[1]
    table.index = ["alpha", "beta"]
    table = table.iloc[:,[0,2,3]]
    table.columns = ["estimate", "t-stat", "p-value"]
    table.index.name = "coefficient"
    return table.round(3)
