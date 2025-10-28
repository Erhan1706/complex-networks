import matplotlib.pyplot as plt
import pickle

def plot_rmse(rmse_series, series_labels, title='RMSE over Time', xlabel='Time Step', ylabel='RMSE',
              tick_distance=500, smoothing_window=None):

    plt.figure(figsize=(10, 6))
    for rmse_values, label in zip(rmse_series, series_labels):
        if smoothing_window is not None:
            smoothed_values = []
            for i in range(len(rmse_values)):
                start_idx = max(0, i - smoothing_window + 1)
                window_values = rmse_values[start_idx:i + 1]
                smoothed_values.append(sum(window_values) / len(window_values))
            plt.plot(smoothed_values, linestyle='-', label=label)
        else:
            plt.plot(rmse_values, linestyle='-', label=label)
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    rmse_lengths = [len(rmse) for rmse in rmse_series]
    max_len = max(rmse_lengths)
    all_ticks = max_len
    ticks = [t for t in range(0, all_ticks, tick_distance)] + [all_ticks]
    plt.xticks(ticks)
    plt.legend()
    plt.show()
    plt.savefig('rmse_plot.png')



if __name__ == "__main__":
    # Example usage
    #rmse_data_1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    #rmse_data_2 = [1.0, 0.85, 0.75, 0.65, 0.55]
    #plot_rmse([rmse_data_1, rmse_data_2], ['Model A', 'Model B'])
    pass