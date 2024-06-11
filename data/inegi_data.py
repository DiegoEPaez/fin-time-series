from pandas.tseries.offsets import MonthEnd, QuarterEnd
from datetime import datetime
import pandas as pd
import logging
import requests

INEGI_TOKEN = '67949ef3-72c7-bdb7-5474-d45f3a41fbb5'


def query_inegi(series):
    """
    Queries "INEGI" API to get specified series (for example CETES rate 28 days) in time
    frame given
    :param series:     series to query for
    :param start_date: begin date for query
    :param end_date:   end date for query
    :return: a JSON object built from API response
    """

    url_1 = 'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/'
    url_2 = '/es/00000/false/BIE/2.0/'
    url_3 = '?type=json'

    logging.info("Querying INEGI...")

    url = url_1 + str(series) + url_2 + INEGI_TOKEN + url_3 

    response = requests.get(url)

    if response.status_code == 200:

        logging.debug("Obtained response from INEGI")

        data = response.json()
        return data
    else:

        logging.debug("Response status code was not successful from Banxico API")
        return None


def format_date(curr_date, prev_date, freq):
    if freq == '6':
        # Quarterly
        if int(curr_date[6:]) == 1:
            curr_date = curr_date[:5] + '01/01'
        elif int(curr_date[6:]) == 2:
            curr_date = curr_date[:5] + '04/01'
        elif int(curr_date[6:]) == 3:
            curr_date = curr_date[:5] + '07/01'
        else:
            curr_date = curr_date[:5] + '10/01'

        report_date = datetime.strptime(curr_date, "%Y/%m/%d") + QuarterEnd(1)
    elif freq == '8':
        curr_date = curr_date + '/01'
        report_date = datetime.strptime(curr_date, "%Y/%m/%d") + MonthEnd(1)
    elif freq == '9':
        if not prev_date or prev_date != curr_date:
            curr_date = curr_date + '/15'
            report_date = datetime.strptime(curr_date, "%Y/%m/%d")
        else:
            curr_date = curr_date + '/01'
            report_date = datetime.strptime(curr_date, "%Y/%m/%d") + MonthEnd(1)

    return report_date


def dwld_inegi(series):
    """
    Dwld Government Bonds series (rates, terms) from INEGI API starting at given
    :return:  df with data
    """
    logging.info("Downloading data from INEGI for series " + str(series))
    data = query_inegi(series)

    if not data:
        return None

    idata = data['Series'][0]

    # Current series name
    curr_series = idata['INDICADOR']
    freq = idata['FREQ']

    # Loop through data for current series
    data_builder = {}
    prev_date = None
    for x in idata['OBSERVATIONS']:
        curr_date = x['TIME_PERIOD']
        report_date = format_date(curr_date, prev_date, freq)
        value = x['OBS_VALUE']
        prev_date = curr_date

        data_builder[report_date] = float(value)

    df = pd.DataFrame.from_dict(data_builder, orient='index')
    df.columns = [curr_series]

    return df

