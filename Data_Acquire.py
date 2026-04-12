import requests
import pandas as pd
from datetime import datetime, timedelta

class Data_Acquisition:

    def __init__(self, url):
        self.url = url

    # -----------------------------
    # Get Data Between Dates (Daily)
    # -----------------------------
    def Get_Data_By_Date_Range(self, start_date, end_date):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []

        current = start

        while current <= end:

            date_str = current.strftime("%Y-%m-%d")

            params = {"date": date_str}

            response = requests.get(self.url, params=params)
            data = response.json()

            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

            current += timedelta(days=1)

        df = pd.DataFrame(all_data)

        return df