import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def do_MLP(dataset, cutpoint):
    # from series of values to windows matrix
    def create_dataset(ds, lb=1):
        dataX, dataY = [], []
        for i in range(len(ds) - lb):
            a = ds[i:(i + lb), 0]
            dataX.append(a)
            dataY.append(ds[i + lb, 0])
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
