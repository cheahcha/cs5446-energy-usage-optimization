from enum import Enum
import pandas as pd
import numpy as np
from prophet import Prophet
from constants import REGION_LABEL, REGION_CODE, State


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
            model = Prophet(yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            seasonality_mode='multiplicative')
            # Add SG holidays
            model.add_country_holidays(country_name='SG')
            model.fit(df_ts)
            self.predictors[region] = model

    def predict(self, region: str, dt_str: str):
        """Predicts electricity consumption and encodes consumption levels to states 
        based on main region category.

        0: "residential",
        1: "commercial",
        2: "industrial",
        3: "others",

        If region is residential and energy consumption is low
        Args:
            region (str): Region
            dt_str (str): Year-Month

        Returns:
            int: State based on region category and consumption level
        """
        region_breakdown = REGION_LABEL[region]
        # Get the main region category
        region_type = max(region_breakdown, key=region_breakdown.get)
        forecast = self.predictors[region].predict(pd.DataFrame({
            'ds': pd.to_datetime([dt_str])
        }))

        consumption = forecast[["yhat"]].values[0][0]

        print(f"{region} {dt_str} -> {consumption} {self.thres[region]}")
        
        state = -1
        # low consumption
        if consumption < self.thres[region][0]: 
            state = REGION_CODE[region_type] * 3 + State.LOW.value
        # high consumption
        elif consumption > self.thres[region][1]: 
            state = REGION_CODE[region_type] * 3 + State.HIGH.value
        # sufficient
        else: 
            state = REGION_CODE[region_type] * 3 + State.MEDIUM.value
        print(f"Primary region type: {region_type} -> State: {state}")

        return state
