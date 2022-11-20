import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymssql
import statsmodels.formula.api as smf
from pandas_datareader import DataReader as pdr
from datetime import datetime, timedelta
from joblib import load

def Ranks(model):
    """
    Keyword arguments:
    model -- should be the pathname of a scikit-learn model saved with joblib.dump. 
    Example:
    Ranks("files/model4.joblib")
    The model must be based on the following features in this order:
        acc, 
        agr, 
        beta, 
        bm, 
        ep, 
        gma, 
        idiovol, 
        lev, 
        mom12m, 
        mom1m, 
        mve, 
        operprof, 
        roeq
    The feature data is pulled from the Rice SQL database, and the function will not run
    unless you are on the Rice network or VPN.  The function returns a pandas series of 
    ranks 1=best, 2=second best, ... indexed by tickers.  
    """
    server = 'fs.rice.edu'
    database = 'stocks'
    username = 'stocks'
    password = '6LAZH1'
    string = "mssql+pymssql://" + username + ":" + password + "@" + server + "/" + database 
    conn = create_engine(string).connect()
    print("made connection")

    ##############################################
    #
    #  Calculate roeq from quarterly reports
    #
    ###############################################

    quarterly = pd.read_sql(
        """
        select datekey as date, ticker, netinc, equity
        from sf1
        where dimension='ARQ' and equity>0 and reportperiod>='2022-01-01'
        order by ticker, datekey
        """,
        conn
    )
    quarterly = quarterly.dropna()
    quarterly["equitylag"] = quarterly.groupby("ticker").equity.shift()
    quarterly["roeq"] = quarterly.netinc / quarterly.equitylag
    quarterly = quarterly.groupby("ticker").last()
    quarterly = quarterly[["roeq"]]
    print("finished roeq")
    print(quarterly.count(), "\n")

    ###################################################
    #
    # Calculate predictors from annual reports
    #
    ###################################################

    annual = pd.read_sql(
        """
        select datekey as date, ticker, netinc, ncfo, assets, assetsavg, equity,
        equityavg, revenue, cor, liabilities, marketcap, sgna, intexp, sharesbas
        from sf1
        where dimension='ARY' and assets>0 and equity>0 and reportperiod>='2020-01-01'
        order by ticker, datekey
        """,
        conn
    )
    annual = annual.dropna(subset=["ticker"])
    annual["equitylag"] = annual.groupby("ticker").equity.shift()
    annual["assetslag"] = annual.groupby("ticker").assets.shift()
    annual["acc"] = (annual.netinc - annual.ncfo) / annual.assetsavg
    annual["agr"] = annual.groupby("ticker").assets.pct_change()
    annual["bm"] = annual.equity / annual.marketcap
    annual["ep"] = annual.netinc / annual.marketcap
    annual["gma"] = (annual.revenue-annual.cor) / annual.assetslag
    annual["lev"] = annual.liabilities / annual.marketcap
    annual["operprof"] = (annual.revenue-annual.cor-annual.sgna-annual.intexp) / annual.equitylag
    annual = annual.groupby("ticker").last()
    annual = annual[["acc", "agr", "bm", "ep", "gma", "lev", "operprof"]]
    print("finished annual predictors")
    print(annual.count(), "\n")
    
    #######################################################
    #
    # Daily data
    #
    #######################################################
    
    start = datetime.today() - timedelta(weeks=156)
    start = start.strftime("%Y-%m-%d")
    string = "select ticker, date, closeadj from sep where date>="
    string += "'" + start + "'"
    string += "order by ticker, date"
    prices = pd.read_sql(string, conn)
    prices = prices.dropna()
    prices["date"] = pd.to_datetime(prices.date)
    print("got daily prices")
    print(prices.count(), "\n")
    
    ##########################################################
    #
    # Weekly returns
    #
    ##########################################################
    
    prices["year"] = prices.date.apply(lambda x: x.isocalendar()[0])
    prices["week"] = prices.date.apply(lambda x: x.isocalendar()[1])
    week = prices.groupby(["year", "week"]).date.max()
    week.name = "weekdate"
    prices = prices.merge(week, on=["year", "week"])
    weekly = prices.groupby(["ticker", "weekdate"]).last()
    returns = weekly.groupby("ticker").closeadj.pct_change()
    returns = returns.reset_index()
    returns.columns = ["ticker", "date", "ret"]
    print("computed weekly returns")
    print(returns.count(), "\n")
    
    ##############################################################
    #
    # Beta and idiosyncratic volatility
    #
    ##############################################################
    
    factors = pdr("F-F_Research_Data_Factors_weekly", "famafrench", start=2019)[0] / 100
    returns = returns.merge(factors, left_on="date", right_on="Date")
    returns["ret"] = returns.ret - returns.RF
    returns["mkt"] = returns["Mkt-RF"]
    returns = returns[returns.date >= "2019-10-01"].dropna()
    
    def regr(d):
        if d.shape[0] < 52:
            return pd.Series(np.nan, index=["beta", "idiovol"])
        else:
            model = smf.ols("ret ~ mkt", data=d)
            result = model.fit()
            beta = result.params["mkt"]
            idiovol = np.sqrt(result.mse_resid)
            return pd.Series([beta, idiovol], index=["beta", "idiovol"])
            
    regression = returns.groupby("ticker").apply(regr)
    regression = regression.groupby("ticker").last()
    print("computed betas and idiosyncratic vols")
    print(regression.count(), "\n")
    
    ################################################################
    #
    # Momentum
    #
    ################################################################
    
    prices = prices[prices.date>="2021-11-01"]
    prices["price12m"] = prices.groupby("ticker").closeadj.shift(253)
    prices["price1m"] = prices.groupby("ticker").closeadj.shift(22)
    prices["price1d"] = prices.groupby("ticker").closeadj.shift(1)
    prices["mom12m"] = prices.price1m / prices.price12m - 1
    prices["mom1m"] = prices.price1d / prices.price1m - 1
    momentum = prices[["ticker", "date", "mom12m", "mom1m"]]
    momentum = momentum.groupby("ticker").last()
    momentum = momentum[["mom12m", "mom1m"]]
    print("computed momentum variables")
    print(momentum.count(), "\n")
    
    ####################################################################
    #
    # Market cap
    #
    #####################################################################
    
    x = pd.read_sql("select max(date) from daily", conn)
    maxdate= str(x.iloc[0, 0])
    string = "select ticker, marketcap from daily where date=" + "'" + maxdate + "'"
    mktcap = pd.read_sql(string, conn)
    mktcap = mktcap.dropna()
    mktcap = mktcap.set_index("ticker")
    mktcap["mve"] = np.log(mktcap.marketcap)
    print("got market cap")
    print(mktcap.count(), "\n")
    
    #######################################################################
    #
    # Merge, load model, predict, and rank
    #
    #######################################################################
    
    df = pd.concat((quarterly, annual, regression, momentum, mktcap), axis=1)
    print(df.shape[0])
    string = "acc, agr, beta, bm, ep, gma, idiovol, lev, mom12m, mom1m, mve, operprof, roeq"
    features = string.split(", ")
    X = df[features].dropna()
    print(X.shape[0])
    mod = load(model)
    predict = pd.Series(mod.predict(X), index=X.index)
    print("finished ranks", predict.shape[0])
    conn.close()
    return predict.rank(ascending=False)

if __name__ == "__main__":
    import sys
    ranks = Ranks(sys.argv[1])
    ranks.to_csv(sys.argv[2])
