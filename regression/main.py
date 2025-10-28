import pickle
from graphing import plot_rmse
from regression.true_lasso import TrueLasso


with open('true_lasso_checkpoint_t3500.pkl', 'rb') as f:
    lasso_model = pickle.load(f)

with open('rmses.pkl', 'rb') as f:
    rmses = pickle.load(f)

plot_rmse([lasso_model.rmses, rmses], ['True Lasso', 'first try'], title='True Lasso RMSE over Time',
          smoothing_window=10)

