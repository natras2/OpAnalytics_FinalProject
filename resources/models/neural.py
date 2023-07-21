import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


def do_MLP(dataset, cutpoint):
    # from series of values to windows matrix
    def create_dataset(dataset, lb=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - lb):
            a = dataset[i:(i + lb), 0]
            dataX.append(a)
            dataY.append(dataset[i + lb, 0])
        return np.array(dataX), np.array(dataY)

    # for reproducibility
    np.random.seed(550)

    # needed for MLP input
    dataset = dataset.values.astype('float32')
    dataset = dataset.reshape(-1, 1)
    dataset = np.hstack((dataset, np.empty((dataset.shape[0], 1))))

    train = dataset[0:cutpoint, :]
    test = dataset[cutpoint:len(dataset), :]

    # sliding window matrices (look_back = window width); dim = n - look_back - 1
    look_back = 42
    testdata = np.concatenate((train[-look_back:], test))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(testdata, look_back)

    # Multilayer Perceptron model
    loss_function = 'mean_squared_error'
    model = Sequential()
    model.add(Dense(64, input_dim=look_back, activation='relu'))  # 1.5n input nodes - Kasstra, Boyd '96
    model.add(Dense(32, input_dim=64, activation='relu'))  # 0.5n input nodes - Kang, '91
    model.add(Dense(1))  # 1 output neuron
    model.compile(loss=loss_function, optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

    # generate predictions for training and forecast for plotting
    trainPredict = model.predict(trainX)
    testForecast = model.predict(testX)

    return trainPredict.squeeze(), testForecast.squeeze()


def do_LSTM(trainset, testset):
    train = pd.Series(np.log(trainset))
    test = pd.Series(np.log(testset))

    # ------------------------------------------------- neural forecast
    scaler = MinMaxScaler()
    scaler.fit_transform(train.values.reshape(-1, 1))
    scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
    scaled_test_data = scaler.transform(test.values.reshape(-1, 1))

    n_input = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

    lstm_model = Sequential()
    lstm_model.add(LSTM(20, activation="relu", input_shape=(n_input, n_features), dropout=0.05))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(generator, epochs=100)
    lstm_model.summary()

    losses_lstm = lstm_model.history.history['loss']
    plt.xticks(np.arange(0, 21, 1))  # convergence trace

    plt.plot(range(len(losses_lstm)), losses_lstm)
    plt.show()

    lstm_predictions_scaled = list()
    batch = scaled_train_data[-n_input:]
    curbatch = batch.reshape((1, n_input, n_features))  # 1 dim more

    # this code concatenates the last (n-i)_input values with the i predicted values
    # towards an array used to forecast the following item
    for i in range(len(test)):
        # the model predicts the following item
        lstm_pred = lstm_model.predict(curbatch)[0]
        # the forecasted value is appended to the previous predictions
        lstm_predictions_scaled.append(lstm_pred)
        # the forecasted value is appended to the curbatch arrays without the first element
        curbatch = np.append(curbatch[:, 1:, :], [[lstm_pred]], axis=1)

    lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
    yfore = np.transpose(lstm_forecast).squeeze()

    # recostruction
    expdata = np.exp(train)  # unlog
    expfore = np.exp(yfore)
    plt.plot((train + test), label="dataset")
    plt.plot(expdata, label="expdata")
    plt.plot([None for x in expdata] + [x for x in expfore], label="forecast")
    plt.legend()
    plt.show()

    return expfore