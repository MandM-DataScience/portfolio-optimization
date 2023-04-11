import json
from configparser import ConfigParser
import pandas as pd
import requests
import urllib3
import yfinance as yf
import fredapi as fa
from simplejson import JSONDecodeError
from urllib3.exceptions import MaxRetryError
from bs4 import BeautifulSoup
import os

from portfolio_optimization import PORTFOLIO_BASE_DIR
from portfolio_optimization.data_lake.mongodb import upload_fred_category, get_fred_categories_collection, \
    get_fred_series_collection, upload_fred_series_many, get_fred_datasets_collection, upload_fred_datasets, \
    upload_yf_target_datasets, get_yf_target_datasets_collection

parser = ConfigParser()
_ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
fred_api_key = parser.get("fred", "fred_api_key")


def request_with_retries(url, headers=None):

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    resp = None
    max_retry = 5
    retry = 0

    while resp is None and retry < max_retry:
        try:
            if headers is not None:
                resp = requests.get(url, verify=False, headers=headers)
            else:
                resp = requests.get(url, verify=False)
        except:
            print(f"{url} conn err - retry")
        retry += 1
    return resp

def download_from_fred(series, filename, measure):

    fred = fa.Fred(api_key="7631f0c483efe1f85e1f1e25a533ad59")
    df = fred.get_series(series).to_frame()

    df = df.reset_index()
    df.columns = ["Date", "Value"]
    df["Measure"] = measure

    if not df.empty:
        df.to_csv(filename)
    else:
        print("{}: empty DF!".format(series))

def download_yahoo_finance_price_data(ticker, interval="1mo"):

    t = yf.Ticker(ticker)

    df = None
    max_retry = 5
    retry = 0

    while df is None and retry < max_retry:
        try:
            df = t.history(period="max", interval=interval)
            df["ticker"] = ticker
        except:
            print(f"{ticker} conn err - retry # {retry+1}")
        retry += 1

    return df

def get_investing_data(url):
    headers = {
        'accept': 'text/plain, */*; q=0.01',
        'accept-encoding': 'gzip, deflate, utf-8',
        'accept-language': 'en,it-IT;q=0.9,it;q=0.8,en-US;q=0.7',
        'cache-control': 'no-cache',
        'origin': 'https://www.investing.com',
        'pragma': 'no-cache',
        'sec-ch-ua': '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
        'upgrade-insecure-requests': '1',
    }
    response = request_with_retries(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    table = soup.find("table", id="curr_table")
    if table is None:
        table = soup.select_one('table[data-test="historical-data-table"]')

    trs = table.find_all("tr")

    rows = []

    for tr in trs:
        tds = tr.find_all("td")

        if len(tds) > 0:
            for i, td in enumerate(tds):
                # print(i, td.text)
                if i == 0:
                    d_ = td.text
                if i == 1:
                    v_ = td.text
            rows.append({"Date": d_, "Price": v_})

    df = pd.DataFrame(rows)
    return df

def get_root_category(api_key):
    data = explore_related(0, api_key)
    if "categories" in data:
        for d in data["categories"]:
            d["_id"] = d.pop("id")
            upload_fred_category(d)

def explore_children(category_id, api_key):
    resp = request_with_retries(
        f"https://api.stlouisfed.org/fred/category/children?category_id={category_id}&api_key={api_key}&file_type=json")
    return resp.json()

def explore_related(category_id, api_key):
    resp = request_with_retries(
        f"https://api.stlouisfed.org/fred/category/related?category_id={category_id}&api_key={api_key}&file_type=json")
    return resp.json()

def explore_series(category_id, api_key):
    url = f"https://api.stlouisfed.org/fred/category/series?category_id={category_id}&api_key={api_key}&file_type=json"
    # print(url)
    resp = request_with_retries(url)
    return resp.json()

def retrieve_series(series_id, api_key, start_date="1776-07-04", end_date="9999-12-31"):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&observation_start={start_date}&observation_end={end_date}&api_key={api_key}&file_type=json"
    # print(url)
    resp = request_with_retries(url)
    return resp.json()

def retrieve_series_metadata(series_id, api_key):
    url = f"https://api.stlouisfed.org/fred/series?series_id={series_id}&api_key={api_key}&file_type=json"
    # print(url)
    r = request_with_retries(url)
    return r.json()['seriess'][0]

def get_oecd_live_dataset():
    url = f'https://stats.oecd.org/SDMX-JSON/data/DP_LIVE/USA..../all?dimensionAtObservation=allDimensions'
    r = requests.get(url)
    data = {"_id": "DP_LIVE", "metadata": {"dataset_id": "DP_LIVE"}, "dataset": r.json()}
    return data

def get_category_tree(api_key):
    cat = get_fred_categories_collection()
    cat = cat.find({}, {"_id": 1})

    already_inserted = []
    already_requested = []

    for c in cat:
        already_inserted.append(c["_id"])

    print("start:", already_inserted)

    for c in already_inserted:

        if c not in already_requested:
            # data = explore_children(c, api_key)
            data = explore_related(c, api_key)
            already_requested.append(c)

            if "categories" in data:
                for d in data["categories"]:
                    d["_id"] = d.pop("id")
                    print(d["_id"])
                    if d["_id"] not in already_inserted:
                        upload_fred_category(d)
                        print("upload", d)
                        already_inserted.append(d["_id"])

def get_series_from_categories(api_key):

    last_category_requested = None

    while True:
        cat = get_fred_categories_collection()
        cat = list(cat.find({}, {"_id": 1}))
        ser = get_fred_series_collection()
        ser = ser.find({}, {"_id": 1})

        already_inserted_series = []
        already_requested_categories = []

        if last_category_requested is not None:
            already_requested_categories = [x["_id"] for x in cat]
            already_requested_categories = \
                already_requested_categories[:already_requested_categories.index(last_category_requested) + 1]

        print(already_requested_categories)
        to_insert = []

        for s in ser:
            already_inserted_series.append(s["_id"])

        for c in cat:
            if c["_id"] not in already_requested_categories:
                try:
                    data = explore_series(c["_id"], api_key)
                except JSONDecodeError:
                    pass
                already_requested_categories.append(c)

                if "seriess" in data and len(data["seriess"]) > 0:

                    for s in data["seriess"]:
                        s["_id"] = s.pop("id")
                        if s["_id"] not in already_inserted_series:
                            to_insert.append(s)
                            already_inserted_series.append(s["_id"])

                print(c["_id"], ", series:", len(to_insert))
                if len(to_insert) > 30000:
                    last_category_requested = c["_id"]
                    break

        upload_fred_series_many(to_insert)

def get_series_datasets(api_key, min_popularity=10):
    ser = get_fred_series_collection()
    ser = list(ser.find({"popularity":{"$gt":min_popularity}}, {"_id": 1}))
    fred = get_fred_datasets_collection()
    fred = fred.find({}, {"_id": 1})

    already_inserted = []
    for f in fred:
        already_inserted.append(f["_id"])

    for s in ser:
        if s["_id"] not in already_inserted:
            try:
                data = retrieve_series(s["_id"], api_key)
                data["_id"] = s["_id"]
                print("upload", s["_id"])
                upload_fred_datasets(data)
            except JSONDecodeError:
                pass
            already_inserted.append(s["_id"])

def get_yf_target_datasets():
    yf_tickers = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^SP500BDT", "^TNX", "^DJUSRE", "USCI", "^DJCI", "GC=F", "CL=F"]

    yf_datasets = get_yf_target_datasets_collection()
    yf_datasets = yf_datasets.find({}, {"_id": 1})

    already_inserted = []
    for y in yf_datasets:
        already_inserted.append(y["_id"])

    for ticker in yf_tickers:
        if ticker not in already_inserted:
            df = download_yahoo_finance_price_data(ticker)
            records = df.reset_index().to_json(orient="records")
            _js = {}
            _js["data"] = json.loads(records)
            _js["_id"] = ticker
            upload_yf_target_datasets(_js)

def retry_get_fred_datasets_with_errors(api_key):
    fred = get_fred_datasets_collection()
    errors = fred.find({"error_code":{"$exists":1}}, {"_id": 1})
    for e in errors:
        try:
            data = retrieve_series(e["_id"], api_key)
            data["_id"] = e["_id"]

            print("upload", e["_id"])
            # DELETE OLD
            fred.delete_one({"_id":e["_id"]})
            # INSERT NEW
            upload_fred_datasets(data)

        except JSONDecodeError:
            pass
