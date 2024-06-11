# Stopped developing because pandas-ta has all this techincal indicators and RSI coincides exactly with that of pandas-ta

def rsi(df, symbol, period=14, avg_type='ewma'):
    """
    Calculate Relative Strength Index. A clear explanation can be found at:
    https://en.wikipedia.org/wiki/Relative_strength_index

    Note that some places use simple moving average to obtain average gains and average losses, however, more serious
    calculations use exponentially weighted moving average, which is the default type.

    :param df: DataFrame to obtain rsi from
    :param symbol: column from DataFrame to use
    :param period: period to use to calculate RSI (usually 14)
    :param avg_type: how to average either 'sma' or 'ewma' (simple moving average or exponentially weighted ...)
    :return:
    """
    # Calculate the price change for each period
    delta = df[symbol].diff()

    # Calculate the average gain and average loss for the specified period
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    if avg_type == 'sma':
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
    else:
        avg_gain = gain.ewm(alpha=1./period).mean()
        avg_loss = loss.ewm(alpha=1./period).mean()

    # Calculate the Relative Strength (RS) by dividing the average gain by the average loss
    rs = avg_gain / avg_loss

    # Calculate the Relative Strength Index (RSI)
    rsi_score = 100 - (100 / (1 + rs))

    return rsi_score
