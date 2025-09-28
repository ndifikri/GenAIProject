import pandas as pd
import numpy as np
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from prophet import Prophet
from pydantic import BaseModel, Field
from typing import List
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

def GenerateDataSales(current_date):
    # Set the start and end dates for one year
    start_date = current_date.replace(year=current_date.year - 2)
    end_date = current_date

    # Generate a date range for each day
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # List of example SKUs
    skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']

    # Create a list to hold the daily sales data
    data = []

    # Generate random sales data and stock data for each SKU for each day
    for date in dates:
        for sku in skus:
            # Generate a random number of units sold between 0 and 100
            units_sold = np.random.randint(0, 101)
            # Generate a random number for stock available between 50 and 200
            stock_available = np.random.randint(300, 2001)
            data.append([date, sku, units_sold, stock_available])

    # Create a pandas DataFrame from the generated data
    sales_df = pd.DataFrame(data, columns=['date', 'sku', 'units_sold', 'current_stock'])
    return sales_df

def GetMonthlySales(sales_df):
# Extract month and year for monthly grouping
    sales_df['month_year'] = sales_df['date'].dt.to_period('M')
    # Group by month_year and sku, then sum units_sold
    monthly_sales = sales_df.groupby(['month_year', 'sku'])['units_sold'].sum().reset_index()
    # Convert month_year back to datetime for plotting
    monthly_sales['month_year'] = monthly_sales['month_year'].dt.to_timestamp()
    return monthly_sales

def GetLatestStock(sales_df):
    latest_stock = sales_df.sort_values('date', ascending=False).groupby('sku').head(1).reset_index(drop=True)
    # Select the desired columns
    sku_stock = latest_stock[['sku', 'current_stock']]
    return sku_stock

def GetForecast(df, x, y, periods):
    result_forecast = []
    for sku_id in df['sku'].unique():
        data = df[df['sku'] == sku_id].reset_index(drop=True)
        data = data[[x, y]] # Filter data for the current SKU and select necessary columns
        # Group by date and sum units_sold, ensuring we have one entry per date for Prophet
        data = data.groupby(x).sum().reset_index()
        data.columns = ['ds', 'y']
        model = Prophet()
        model.fit(data)
        df_future = model.make_future_dataframe(periods=periods)
        forecast_prophet = model.predict(df_future)
        # Extract forecast values and convert to list of strings
        forecast_value = forecast_prophet['yhat'][-periods:].round().tolist() # Convert to list
        result = {'sku': sku_id, 'forecast': [int(x) for x in forecast_value]} # Convert to list of strings
        result_forecast.append(result)
    return result_forecast

def GetForecastDF(json_forecast):
  # Create a list to store the data for the new DataFrame
  forecast_summary_data = []
  # Iterate through the forecast results
  for item in json_forecast:
      sku_name = item['sku']
      # Convert forecast values from strings to integers and sum them
      total_forecast = sum(int(value) for value in item['forecast'])
      forecast_summary_data.append({'sku': sku_name, 'forecast_30_days': total_forecast})
  # Create a DataFrame from the summary data
  forecast_summary_df = pd.DataFrame(forecast_summary_data)
  return forecast_summary_df

def GetExpectedDay(stok_sisa, forecast):
    hari_ke = 0
    for permintaan_harian in forecast:
        hari_ke += 1
        stok_sisa -= permintaan_harian
        if stok_sisa <= 0:
            return hari_ke
    return hari_ke

def GetRemainDays(latest_stock_wh, forecast_wh):
    results = []
    for sku_id in latest_stock_wh['sku']:
        s = latest_stock_wh[latest_stock_wh['sku'] == 'SKU005']['current_stock'].values
        f = next(item for item in forecast_wh if item["sku"] == sku_id)['forecast']
        remain_days = GetExpectedDay(s, f)
        result = {
            "sku" : sku_id,
            "remain_days_to_out_of_stock" : remain_days
        }
        results.append(result)
    return pd.DataFrame(results)

def GetQuantityNeeded(combined_df):
    class ResponseFormatter(BaseModel):
        """Always use this format to give estimation quantity needed of items."""
        sku: str = Field(description="SKU ID of the item.")
        quantity_needed: int = Field(description="estimation quantity needed of item.")
    
    class ResponWrapper(BaseModel):
        list_item: List[ResponseFormatter]
    
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash'
    )
    data = combined_df.to_dict(orient='records')
    model_with_structure = llm.with_structured_output(ResponWrapper)
    prompt = f'''You are an Warehouse Optimization Expert, your task is calculate quantity need for each SKU based on given data.
Here is detail about data:
-  sku : SKU ID of the item.
-  current_stock : number of current item stock.
-  forecast_30_days : forecast for next 30 days quantity needs.
-  remain_days_to_out_of_stock : number of days that item will out of stock.
Here is the data:
{data}
'''
    response = model_with_structure.invoke(prompt)
    result = response.model_dump()
    df_result = pd.DataFrame(result["list_item"])
    return df_result

def GetMaxDayToRestock(df):
    df['calculated_days'] = df['remain_days_to_out_of_stock'] - 8

    # 2. Tentukan Kondisi dan Pilihan Nilai
    conditions = [
        # Kondisi 1: quantity_needed <= 0
        df['quantity_needed'] <= 0,

        # Kondisi 2: quantity_needed > 0 DAN hasil perhitungan <= 0 (yang harus diisi 'immediately')
        (df['quantity_needed'] > 0) & (df['calculated_days'] <= 0)
    ]

    choices = [
        # Pilihan untuk Kondisi 1: None
        None,

        # Pilihan untuk Kondisi 2: "immediately"
        "immediately"
    ]

    # 3. Terapkan logika menggunakan np.select
    # Default: Jika tidak ada kondisi di atas yang terpenuhi (yaitu quantity_needed > 0 DAN hasil hitungan > 0), gunakan 'calculated_days'.
    df['max_days_to_restock'] = np.select(
        conditions,
        choices,
        default=df['calculated_days']
    )

    # Hapus kolom bantuan 'calculated_days' jika tidak diperlukan
    df = df.drop(columns=['calculated_days'], axis=1)
    return df['max_days_to_restock'].values

current_date = datetime.date.today()

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="Supply Chain Optimization",
    page_icon="📊"
)
st.title("Supply Chain for Warehouse Optimization")
st.header("Assumptions")
st.markdown(f'''
-  Current Date : {current_date}
-  Replenishment Lead Time : 7 (for all items)
-  Sales transaction record generated randomly.
-  Transaction data will be re-generate every refresh/reload apps.
''')

tab1, tab2, tab3 = st.tabs(["Warehouse 1", "Warehouse 2", "Warehouse 3"])
with tab1:
    st.header("Warehouse 1 Information")
    wh1 = GenerateDataSales(current_date)
    monthly_wh1 = GetMonthlySales(wh1)
    latest_stock_wh1 = GetLatestStock(wh1)
    st.write("**Line Chart of Unit Sold by Month**")
    st.line_chart(monthly_wh1, x="month_year", y="units_sold", color="sku")

    if st.button("Forecast WH1"):
        st.write("**Item Stock and Forecast**")
        forecast_wh1 = GetForecast(wh1, 'date', 'units_sold', 30)
        forecast_df_wh1 = GetForecastDF(forecast_wh1)
        combined_df1 = pd.merge(latest_stock_wh1, forecast_df_wh1, on='sku')
        remain_days_df1 = GetRemainDays(latest_stock_wh1, forecast_wh1)
        combined_df1 = pd.merge(combined_df1, remain_days_df1, on='sku')
        get_item_estimation = GetQuantityNeeded(combined_df1)
        combined_df1 = pd.merge(combined_df1, get_item_estimation, on='sku')
        combined_df1["max_days_to_restock"] = GetMaxDayToRestock(combined_df1)
        st.dataframe(combined_df1.drop(columns=['calculated_days'], axis=1))
with tab2:
    st.header("Warehouse 2 Information")
    wh2 = GenerateDataSales(current_date)
    monthly_wh2 = GetMonthlySales(wh2)
    latest_stock_wh2 = GetLatestStock(wh2)
    st.write("**Line Chart of Unit Sold by Month**")
    st.line_chart(monthly_wh2, x="month_year", y="units_sold", color="sku")

    if st.button("Forecast WH2"):
        st.write("**Item Stock and Forecast**")
        forecast_wh2 = GetForecast(wh2, 'date', 'units_sold', 30)
        forecast_df_wh2 = GetForecastDF(forecast_wh2)
        combined_df2 = pd.merge(latest_stock_wh2, forecast_df_wh2, on='sku')
        remain_days_df2 = GetRemainDays(latest_stock_wh2, forecast_wh2)
        combined_df2 = pd.merge(combined_df2, remain_days_df2, on='sku')
        get_item_estimation = GetQuantityNeeded(combined_df2)
        combined_df2 = pd.merge(combined_df2, get_item_estimation, on='sku')
        combined_df2["max_days_to_restock"] = GetMaxDayToRestock(combined_df2)
        st.dataframe(combined_df2.drop(columns=['calculated_days'], axis=1))
with tab3:
    st.header("Warehouse 3 Information")
    wh3 = GenerateDataSales(current_date)
    monthly_wh3 = GetMonthlySales(wh3)
    latest_stock_wh3 = GetLatestStock(wh3)
    st.write("**Line Chart of Unit Sold by Month**")
    st.line_chart(monthly_wh3, x="month_year", y="units_sold", color="sku")
    if st.button("Forecast WH3"):
        st.write("**Item Stock and Forecast**")
        forecast_wh3 = GetForecast(wh3, 'date', 'units_sold', 30)
        forecast_df_wh3 = GetForecastDF(forecast_wh3)
        combined_df3 = pd.merge(latest_stock_wh3, forecast_df_wh3, on='sku')
        remain_days_df3 = GetRemainDays(latest_stock_wh3, forecast_wh3)
        combined_df3 = pd.merge(combined_df3, remain_days_df3, on='sku')
        get_item_estimation = GetQuantityNeeded(combined_df3)
        combined_df3 = pd.merge(combined_df3, get_item_estimation, on='sku')
        combined_df3["max_days_to_restock"] = GetMaxDayToRestock(combined_df3)
        st.dataframe(combined_df3.drop(columns=['calculated_days'], axis=1))