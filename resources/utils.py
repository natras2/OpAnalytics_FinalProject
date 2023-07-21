import numpy as np
import pandas as pd


# calculate logdiff
def logdiff(data):
    log_series = [np.log(x) for x in data]
    logdiff_series = [log_series[i] - log_series[i - 1] for i in range(1, len(log_series))]
    return log_series, logdiff_series


# invert logdiff
def invert_logdiff(first_item, diff_data, is_first_item_log=True):
    first_item = first_item if is_first_item_log else np.log(first_item)
    result = np.concatenate(([first_item], diff_data))
    result = np.cumsum(result)
    result = [np.exp(x) for x in result]
    return result


if __name__ == "__main__":
    dataset = pd.Series([2, 4, 6, 9, 11], index=range(6, 11))
    ls, lds = logdiff(dataset)
    data_reversed_log = invert_logdiff(ls[0], lds)
    data_reversed = invert_logdiff(dataset.iloc[0], lds, False)
