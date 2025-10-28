import matplotlib.pyplot as plt


def plot_rmse(rmse_series, series_labels, title='RMSE over Time', xlabel='Time Step', ylabel='RMSE',
              max_n_ticks=20):

    plt.figure(figsize=(10, 6))
    for rmse_values, label in zip(rmse_series, series_labels):
        plt.plot(rmse_values, linestyle='-', label=label)
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    all_ticks = len(rmse_series[0])
    if all_ticks > max_n_ticks:
        step = all_ticks // max_n_ticks
        plt.xticks(range(0, all_ticks, step))
    else:
        plt.xticks(range(all_ticks))
    plt.legend()
    plt.show()
    plt.savefig('rmse_plot.png')


if __name__ == "__main__":
    # Example usage
    rmse_data_1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    rmse_data_2 = [1.0, 0.85, 0.75, 0.65, 0.55]
    plot_rmse([rmse_data_1, rmse_data_2], ['Model A', 'Model B'])