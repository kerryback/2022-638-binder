import pandas as pd
import plotly.graph_objects as go

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


