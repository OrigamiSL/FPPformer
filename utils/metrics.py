import numpy as np


def MAE(pred, true):
    mae = np.abs(pred - true)
    return np.mean(mae)


def MSE(pred, true):
    mse = (pred - true) ** 2
    return np.mean(mse)


def SMAPE(pred, true):
    return 200 * np.mean(np.abs((pred - true) / (np.abs(true) + np.abs(pred))))


def MASE(pred, true, input_x, frequency):
    denom = np.mean(np.abs(input_x[:, frequency:, :] - input_x[:, :input_x.shape[1] - frequency, :]), axis=1) + 1e-6
    mase = np.mean(np.mean(np.abs(pred - true), axis=1) / denom)
    return mase


def OWA(pred, true, pred_naive, input_x, frequency):
    smape_init = SMAPE(pred, true)
    smape_naive = SMAPE(pred_naive, true)
    mase_init = MASE(pred, true, input_x, frequency)
    mase_naive = MASE(pred_naive, true, input_x, frequency)
    owa = 0.5 * (smape_init / smape_naive + mase_init / mase_naive)
    return owa


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse


def metric_M4(pred, true, pred_naive, input_x, frequency=1):
    smape = SMAPE(pred, true)
    owa = OWA(pred, true, pred_naive, input_x, frequency)
    return smape, owa
