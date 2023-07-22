import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def do_autoARIMA(dataset, fore_periods, is_seasonal: bool, periodicity=0):
    # parameters p,d,q,P,D,Q are not known, but m is
    # we use pmdarima to estimate them
    if is_seasonal:
        model = pm.auto_arima(dataset, start_p=1, start_q=1,
                              test='adf', max_p=3, max_q=3, m=periodicity,
                              start_P=0, seasonal=True,
                              d=None, D=1, trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)  # False full grid
    else:
        model = pm.auto_arima(dataset,
                              test='adf', seasonal=False, d=0,
                              trace=True, error_action='ignore',
                              suppress_warnings=True,
                              stepwise=False)  # False full grid
    print(model.summary())
    sfit = model.fit(dataset)

    yfore = sfit.predict(n_periods=fore_periods)  # forecast
    return yfore


def do_SARIMAX(dataset, periodicity, fore_periods):
    # parameters p,d,q,P,D,Q,m are known
    # these parameters have been estimated through auto_arima
    sarima_model = SARIMAX(dataset, order=(1, 0, 1), seasonal_order=(0, 1, 1, periodicity))
    sfit = sarima_model.fit()

    forewrap = sfit.get_forecast(steps=fore_periods)
    yfore = forewrap.predicted_mean

    return yfore
