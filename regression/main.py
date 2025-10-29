import pickle
from graphing import plot_rmse
from regression.true_lasso import TrueLasso


def graph_files(file_list, labels, is_model, smoothing_window=10):
    rmse_series = []
    for i, file in enumerate(file_list):
        with open(file, 'rb') as f:
            model = pickle.load(f)
            if is_model[i]:
                rmse_series.append(model.rmse_history)
            else:
                if len(model) > 1:
                    for sub_list in model:
                        rmse_series.append(sub_list)
                else:
                    rmse_series.append(model)

    plot_rmse(rmse_series, labels, smoothing_window=smoothing_window)


if __name__ == "__main__":
    graph_files(['rmses.pkl', 'rmses_b7.pkl'], ['small_negativ', 'small_positive', 'small_all',
                                                     'big_negative', 'big_positive', 'big_all'],
                [False, False], smoothing_window=100)
