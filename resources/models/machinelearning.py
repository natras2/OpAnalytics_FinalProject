import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def do_bagging(ds, lookback, cutpoint):
    # populate the dataframe with the time shifts
    dataset = pd.DataFrame()
    for i in range(lookback, 0, -1):
        dataset['t-' + str(i)] = ds.shift(i)

    dataset['t'] = ds.values
    dataset = dataset[lookback:]  # df starts from 0

    x = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    xtrain, xtest = x[:-cutpoint], x[-cutpoint:]
    ytrain, ytest = y[:-cutpoint], y[-cutpoint:]

    # define and fit the model
    RFmodel = RandomForestRegressor(n_estimators=500, random_state=1)
    RFmodel.fit(xtrain, ytrain)

    # forecast testset
    pred = RFmodel.predict(xtest)
    mse = mean_absolute_error(ytest, pred)
    print("RandomForest - MSE = {}".format(mse))

    # Stats about the trees in random forest
    n_nodes = []
    max_depths = []
    for ind_tree in RFmodel.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'RandomForest - Average number of nodes {int(np.mean(n_nodes))}')
    print(f'RandomForest - Average maximum depth {int(np.mean(max_depths))}')

    return pred
