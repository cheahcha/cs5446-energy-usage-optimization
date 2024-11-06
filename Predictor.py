import pandas as pd
import numpy as np
from prophet import Prophet
from utils import REGION_LABEL, REGION_CODE


class Predictor:

    def __init__(self, df_consumption: pd.DataFrame):
        self.df_consumption = df_consumption
        self.df_consumption.replace('s', np.nan, inplace=True)
        self.df_consumption.replace('-', np.nan, inplace=True)

        self.thres = {}
        self._cal_thre()

        self.predictors = {}
        self._train()

    def _cal_thre(self):
        for i in range(1, len(self.df_consumption)):
            region = self.df_consumption.iloc[i]["region"]
            ts = self.df_consumption.iloc[i].T[1:]
            ts.index = pd.to_datetime(ts.index)
            ts = ts.astype(float)

            lower_thre = ts.quantile(1 / 3)
            high_thre = ts.quantile(2 / 3)

            self.thres[region] = [lower_thre, high_thre]

    def _train(self):
        for i in range(1, len(self.df_consumption)):
            region = self.df_consumption.iloc[i]["region"]
            ts = self.df_consumption.iloc[i].T[1:]
            ts.index = pd.to_datetime(ts.index)
            ts = ts.astype(float)
            df_ts = pd.DataFrame({
                'ds': ts.index,
                'y': ts.values
            })
            model = Prophet()
            model.fit(df_ts)
            self.predictors[region] = model

    def predcit(self, region: str, dt_str: str):
        region_type = REGION_LABEL[region]
        forecast = self.predictors[region].predict(pd.DataFrame({
            'ds': pd.to_datetime([dt_str])
        }))

        consumption = forecast[["yhat"]].values[0][0]

        print(f"{region} {dt_str} -> {consumption} {self.thres[region]}")

        if consumption < self.thres[region][0]:
            return REGION_CODE[region_type] * 3
        elif consumption > self.thres[region][1]:
            return REGION_CODE[region_type] * 3 + 2
        else:
            return REGION_CODE[region_type] * 3 + 1
