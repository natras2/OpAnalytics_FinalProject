import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from resources.utils import logdiff, invert_logdiff
from resources.models.neural import do_MLP
from resources.models.statistical import do_SARIMAX, do_autoARIMA

if __name__ == "__main__":
    original_df = pd.read_csv("dataset/IoT_traffic_management.csv")

    # Preprocessing
    original_df = original_df.drop(original_df[original_df.Junction > 1].index)
    original_df['Date'] = pd.to_datetime(original_df['DateTime']).dt.date
    original_df['Time'] = pd.to_datetime(original_df['DateTime']).dt.time
    original_df = original_df[['Date', 'Time', 'Vehicles']]

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

    # Data Visualization
    print("Rows: ", df.shape[0])
    print("Columns: ", df.shape[1])
    print("Missing values:\n", df.isnull().any())

    # Slicing in train and test sets
    dataset = df.Vehicles

    cutpoint = int(len(dataset) * 0.7)
    train = dataset[:cutpoint]
    test = dataset[cutpoint:]

    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.xlabel('time')
    plt.ylabel('n of cars')
    plt.legend()
    plt.show()

    # The "period" defines the number of periods within a season.
    # As I wanted to exploit a weekly seasonality I calculated 6 records per day * 7 days per week = 42 periods
    n_periods = 42

    # To perform the decomposition
    decomposition = seasonal_decompose(dataset, model='multiplicative', period=n_periods)
    decomposition.plot()
    plt.show()

    # Not-stationary dataset: SARIMAX - Statistical prediction
    sarimax_fore = do_SARIMAX(train, n_periods, len(test))
    sarimax_fore = pd.Series(sarimax_fore, index=range(len(train), len(train) + len(sarimax_fore)))

    plt.plot(train, label="train")
    plt.plot(test, label="expected", color="darkgray")
    plt.plot(sarimax_fore, label="forecast", alpha=0.5)
    plt.legend()
    plt.show()

    # Stationary dataset: SARIMAX - Statistical prediction
    # l_dataset, ld_dataset = logdiff(dataset)
    #
    # ld_train = ld_dataset[:cutpoint]
    # ld_test = ld_dataset[cutpoint:]
    #
    # ld_sarimax_fore = do_autoARIMA(ld_train, len(ld_test), False)
    #
    # sarimax_fore = invert_logdiff(test.iloc[0], ld_sarimax_fore, False)
    # sarimax_fore = pd.Series(sarimax_fore, index=range(len(train), len(train) + len(sarimax_fore)))
    #
    # plt.plot(train, label="train")
    # plt.plot(test, label="expected", color="darkgray")
    # plt.plot(sarimax_fore, label="forecast", alpha=0.5)
    # plt.legend()
    # plt.show()

    # Not-stationary dataset: MLP - Neural prediction
    MLP_train_pred, MLP_fore = do_MLP(dataset, cutpoint)
    MLP_fore = pd.Series(MLP_fore, index=range(len(train), len(train) + len(MLP_fore)))

    plt.plot(train, label="train")
    plt.plot(test, label="expected", color="darkgray")
    plt.plot(MLP_fore, label="forecast", alpha=0.5)
    plt.legend()
    plt.show()


    pass