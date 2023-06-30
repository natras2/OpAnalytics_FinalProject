import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
from statsmodels.tsa.seasonal import seasonal_decompose

if __name__ == "__main__":
    original_df = pd.read_csv("./datasets/IoT_traffic_management.csv")

    # Preprocessing
    original_df = original_df.drop(original_df[original_df.Junction > 1].index)
    original_df['Date'] = pd.to_datetime(original_df['DateTime']).dt.date
    original_df['Time'] = pd.to_datetime(original_df['DateTime']).dt.time
    original_df = original_df[['Date', 'Time', 'Vehicles']]

    # Exctraction of the last two months
    #original_df['Date'] = pd.to_datetime(original_df['Date'], errors='coerce')
    #original_df = original_df.drop(original_df[original_df['Date'].dt.year < 2017].index)
    #original_df = original_df.drop(original_df[original_df['Date'].dt.month < 5].index)

    # Extraction of the last 9 weeks of data
    # 24 hours * 7 days * 9 weeks = 1512 records
    original_df = original_df.tail(1512)

    i = 0
    df = pd.DataFrame(columns=['Date', 'Time', 'Vehicles'])
    for x in range(0, len(original_df), 4):
        vehicles = np.mean(original_df.iloc[x:x+4, 2].to_numpy())
        df.loc[i] = [
            original_df.iloc[x, 0],
            str(x % 24) + " - " + str((x + 4) % 24),
            vehicles
        ]
        i += 1

    # Slicing in train and test sets
    dataset = df.Vehicles

    testlength = int(len(dataset) * 0.30)
    train = dataset[:-testlength]
    test = dataset[-testlength:]

    plt.plot(train, label="train")
    plt.plot([None for i in train] + [x for x in test], label="test")
    plt.xlabel('time')
    plt.ylabel('n of cars')
    plt.legend()
    plt.show()

    # To perform the decomposition
    # "period" paramenter defines the number of periods within a season.
    # As I wanted to exploit a weekly seasonality I calculated 6 records per day * 7 days per week = 42 periods
    decomposition = seasonal_decompose(dataset, model='multiplicative', period=42)
    decomposition.plot()
    plt.show()

    # calculate logdiff
    def logdiff(data, interval):
        result = [log(x) for x in data]
        result = [result[i] - result[i - interval] for i in range(interval, len(result))]
        return result

    # invert logdiff
    def invert_logdiff(orig_data, diff_data, interval):
        orig_data = [log(x) for x in orig_data]
        result = [diff_data[i - interval] + orig_data[i - interval] for i in range(interval, len(orig_data))]
        result = [exp(x) for x in result]
        return result


    ds_logdiff = logdiff(dataset, 1)
    plt.plot(ds_logdiff)
    plt.show()

    


    pass