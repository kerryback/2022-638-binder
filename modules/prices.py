import pandas as pd
from sqlalchemy import create_engine
import pymssql

def Prices(stocktrak=None):
    """
    Keyword argument (optional):
    stocktrak -- if supplied, should be the pathname of an OpenPosition_date.csv file.
    Example:
    Prices("files/OpenPosition_3_19_2022.csv")
    Returns a pandas series of stock prices indexed by tickers.  Uses the Rice SQL database and
    will not run unless on the Rice network or VPN.  If the stocktrak argument is provided, the
    SQL prices will be updated with prices from the StockTrak csv file.
    """
    server = 'fs.rice.edu'
    database = 'stocks'
    username = 'stocks'
    password = '6LAZH1'
    string = "mssql+pymssql://" + username + ":" + password + "@" + server + "/" + database 
    conn = create_engine(string).connect()
    print("made connection")
    
    #################################################
    #
    # get prices from SQL database
    #
    ##################################################

    x = pd.read_sql("select max(date) from sep", conn)
    maxdate= str(x.iloc[0, 0])
    string = "select ticker, close_ from sep where date=" + "'" + maxdate + "'"
    prices = pd.read_sql(string, conn)
    prices.columns = ["ticker", "price"]
    prices = prices.set_index("ticker")
    prices = prices.squeeze()
    print("got SQL data")
  
    ####################################################
    #
    # use prices from StockTrak file for open positions
    #
    #####################################################

    if stocktrak:
        prices2 = pd.read_csv(stocktrak, index_col="Symbol")
        for tick in prices2.index:
            prices[tick] = prices2.LastPrice.loc[tick]

    print("finished prices")
    conn.close()
    return prices
    

if __name__ == "__main__":
    import sys
    stocktrak = sys.argv[1]
    prices = Prices(stocktrak)
    prices.to_csv(sys.argv[2])

