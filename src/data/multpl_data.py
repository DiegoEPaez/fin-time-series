import pandas as pd
import requests
from bs4 import BeautifulSoup


def query_multpl(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36"
        )
    }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content.decode("utf-8"), features="lxml")
    tr_elems = soup.select("#datatable tr")

    raw_data = [
        [list(td.children)[-1].text.strip() for td in tr_elem.select("td")]
        for tr_elem in tr_elems[1:]
    ]

    return raw_data


def dwld_multpl(ratio):
    """
    Query the multpl.com API for a given ratio.

    Args:
        ratio (str): The ratio to query.

    Returns:
        dict: The response from the API.
    """
    ratio_url = f"https://www.multpl.com/{ratio}/table/by-month"
    ratio_data = query_multpl(ratio_url)

    df_ratio = pd.DataFrame(ratio_data, columns=["date", ratio])
    df_ratio["date"] = pd.to_datetime(df_ratio["date"], format="%b %d, %Y")
    df_ratio[ratio] = df_ratio[ratio].astype("float")
    df_ratio = df_ratio.set_index("date")

    return df_ratio
