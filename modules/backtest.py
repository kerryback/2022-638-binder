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
    
    df["rnk"] = df.groupby("date").predict.rank(method="first", ascending=False)
    best = df[df.rnk<=numstocks]
    df["rnk"] = df.groupby("date").predict.rank(method="first")
    worst = df[df.rnk<=numstocks]

    best_rets = best.groupby("date").ret.mean()
    worst_rets = worst.groupby("date").ret.mean()
    rets = pd.concat((best_rets, worst_rets), axis=1)
    rets.columns = ["best", "worst"]
    return rets

def cumplot(rets):
    traces = []
    for ret in [x for x in rets.columns if x !="date"]:
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
        template="none",
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
    rets = df.copy().set_index("date")
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

    trace1 = go.Scatter(
        x=sds,
        y=mns,
        text=rets.columns.to_list(),
        mode="markers",
        marker=dict(size=10),
        hovertemplate="%{text}<br>mn=%{y:.1%}<br>sd=%{x:.1%}<extra></extra>",
        showlegend=False
    )

    cd = np.empty(shape=(1, 3, 1), dtype=float)
    cd[:, 0] = np.array(w[0])
    cd[:, 1] = np.array(w[1])
    cd[:, 2] = np.array(w[2])
    string = "Tangency portfolio:<br>"
    string += r1 + ": %{customdata[0]:.1%}<br>"
    string += r2 + ": %{customdata[1]:.1%}<br>"
    string += r3 + ": %{customdata[2]:.1%}<br>"
    string += "<extra></extra>"
    trace2 = go.Scatter(
        x=[sd],
        y=[mn],
        mode="markers",
        marker=dict(size=10),
        customdata=cd,
        hovertemplate=string,
        showlegend=False
    )

    x = np.linspace(0, mxsd, 51)
    y = rf+x*(mn-rf)/sd
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        hovertemplate=f"Sharpe ratio = {(mn-rf)/sd:0.1%}<extra></extra>",
        showlegend=False,
    )

    maxmn = np.max(y)
    ms = np.linspace(np.min(mns), maxmn, 51)
    ps = [port(m) for m in ms]
    ss = [np.sqrt(p@cov@p) for p in ps]
    cd = np.empty(shape=(len(ps), 3, 1), dtype=float)
    for i in range(3):
        cd[:, i] = np.array([w[i] for w in ps]).reshape(-1, 1)
    string = r1 + " = %customdata[0]:.1%}<br>" 
    string += r2 + " = %customdata[1]:.1%}<br>"
    string += r3 + " = %customdata[2]:.1%}<br>"
    string += "<extra></extra>"
    trace4 = go.Scatter(
        x=ss,
        y=ms,
        mode="lines",
        customdata=cd,
        hovertemplate=string,
        showlegend=False,
    )

    fig = go.Figure()
    for trace in [trace1, trace2, trace3, trace4]:
        fig.add_trace(trace)
    fig.update_layout(
        template="none",
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        yaxis_title="Annualized Mean",
        xaxis_title="Annualized Standard Deviation",
        xaxis_rangemode="tozero",
        yaxis_rangemode="tozero",
    )
    return fig


