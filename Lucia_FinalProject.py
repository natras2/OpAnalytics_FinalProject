import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from resources.models.machinelearning import do_bagging
from resources.models.neural import do_MLP
from resources.models.statistical import do_SARIMAX  # , do_autoARIMA
# from resources.utils import logdiff, invert_logdiff
from resources.dm_test import dm_test
from sklearn.metrics import mean_absolute_error

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

    # SARIMAX - Statistical prediction
    SARIMAX_fore = do_SARIMAX(train, n_periods, len(test))
    SARIMAX_fore = pd.Series(SARIMAX_fore, index=range(len(train), len(train) + len(SARIMAX_fore)))

    plt.plot(train, label="train")
    plt.plot(test, label="expected", color="darkgray")
    plt.plot(SARIMAX_fore, label="forecast", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('n of cars')
    plt.title("SARIMAX - Statistical prediction")
    plt.legend()
    plt.show()

    # Stationary dataset: SARIMAX - Statistical prediction
    # l_dataset, ld_dataset = logdiff(dataset)
    #
    # ld_train = ld_dataset[:cutpoint]
    # ld_test = ld_dataset[cutpoint:]
    #
    # ld_SARIMAX_fore = do_autoARIMA(ld_train, len(ld_test), False)
    #
    # SARIMAX_fore = invert_logdiff(test.iloc[0], ld_SARIMAX_fore, False)
    # SARIMAX_fore = pd.Series(SARIMAX_fore, index=range(len(train), len(train) + len(SARIMAX_fore)))
    #
    # plt.plot(train, label="train")
    # plt.plot(test, label="expected", color="darkgray")
    # plt.plot(SARIMAX_fore, label="forecast", alpha=0.5)
    # plt.legend()
    # plt.show()

    # MLP - Neural prediction
    MLP_train_pred, MLP_fore = do_MLP(dataset, cutpoint)
    MLP_fore = pd.Series(MLP_fore, index=range(len(train), len(train) + len(MLP_fore)))

    plt.plot(train, label="train")
    plt.plot(test, label="expected", color="darkgray")
    plt.plot(MLP_fore, label="forecast", color="green", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('n of cars')
    plt.title("MLP - Neural prediction")
    plt.legend()
    plt.show()

    # Bagging:RandomForest - Machine Learning prediction
    RF_fore = do_bagging(dataset, n_periods, len(dataset) - cutpoint)
    RF_fore = pd.Series(RF_fore, index=range(len(train), len(train) + len(RF_fore)))

    plt.plot(train, label="train")
    plt.plot(test, label="expected", color="darkgray")
    plt.plot(RF_fore, label="forecast", color="red", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('n of cars')
    plt.title("RandomForest - Machine Learning prediction")
    plt.legend()
    plt.show()

    # Diebold-Mariano test
    dm_SARIMAX_MLP = dm_test(test, SARIMAX_fore, MLP_fore, h=1, crit="MSE")
    dm_SARIMAX_RF = dm_test(test, SARIMAX_fore, RF_fore, h=1, crit="MSE")
    dm_RF_MLP = dm_test(test, RF_fore, MLP_fore, h=1, crit="MSE")

    print(dm_SARIMAX_MLP)
    print(dm_SARIMAX_RF)
    print(dm_RF_MLP)

    # Define the critical values (e.g., for a 95% confidence level)
    critical_value = 1.96
    p_value = 0.025

    comparison_result = pd.DataFrame({
        'SARIMAX': [0, 0, 0],
        'MLP': [0, 0, 0],
        'RF': [0, 0, 0]
        }, index=['SARIMAX', 'MLP', 'RF'])

    # Compare the DM statistics with the critical value and the p-value
    if abs(dm_SARIMAX_MLP.DM) > critical_value and dm_SARIMAX_MLP.p_value < p_value:
        # Null-hypothesis rejected
        if dm_SARIMAX_MLP.DM > 0:
            print("SARIMAX forecast is significantly more accurate than MLP's")
            comparison_result.loc['SARIMAX', 'MLP'] = 1
        else:
            print("MLP forecast is significantly more accurate than SARIMAX's")
            comparison_result.loc['MLP', 'SARIMAX'] = 1

    if abs(dm_SARIMAX_RF.DM) > critical_value and dm_SARIMAX_RF.p_value < p_value:
        # Null-hypothesis rejected
        if dm_SARIMAX_RF.DM > 0:
            print("SARIMAX forecast is significantly more accurate than RandomForest's")
            comparison_result.loc['SARIMAX', 'RF'] = 1
        else:
            print("RandomForest forecast is significantly more accurate than SARIMAX's")
            comparison_result.loc['RF', 'SARIMAX'] = 1

    if abs(dm_RF_MLP.DM) > critical_value and dm_RF_MLP.p_value < p_value:
        # Null-hypothesis rejected
        if dm_RF_MLP.DM > 0:
            print("RandomForest forecast is significantly more accurate than MLP's")
            comparison_result.loc['RF', 'MLP'] = 1
        else:
            print("MLP forecast is significantly more accurate than RandomForest's")
            comparison_result.loc['MLP', 'RF'] = 1

    max_sum_value = comparison_result.sum(axis=1).max()
    if max_sum_value == 2:
        most_accurate_model = comparison_result.sum(axis=1).idxmax()
        print("The most accurate model is {}".format(most_accurate_model))
    else:
        print("There isn't a model that is significantly more accurate than both the other two according to the Diebold-Mariano test")
        print("I proceed to check for the most accurate model using MAE (Mean Absolute Error)")

        # Check for the most accurate model using MAE (Mean Absolute Error)
        SARIMAX_mae = mean_absolute_error(test, SARIMAX_fore)
        print("SARIMAX - MAE = {}".format(SARIMAX_mae))
        comparison_result.loc['SARIMAX', 'SARIMAX'] = SARIMAX_mae

        MLP_mae = mean_absolute_error(test, MLP_fore)
        print("MLP - MAE = {}".format(MLP_mae))
        comparison_result.loc['MLP', 'MLP'] = MLP_mae

        RF_mae = mean_absolute_error(test, RF_fore)
        print("RandomForest - MAE = {}".format(RF_mae))
        comparison_result.loc['RF', 'RF'] = RF_mae

        most_accurate_model = comparison_result.sum(axis=1).idxmin()
        most_accurate_model = "RandomForest" if most_accurate_model == "RF" else most_accurate_model
        print("The most accurate model is {}".format(most_accurate_model))
