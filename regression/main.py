import pickle
from graphing import plot_rmse
from regression.true_lasso import TrueLasso


with open('true_lasso_checkpoint_t3500.pkl', 'rb') as f:
    lasso_model = pickle.load(f)

with open('true_lasso_checkpoint_t3000.pkl', 'rb') as f:
    lasso_model2 = pickle.load(f)

print(lasso_model2.reg_model.coef_)
print(lasso_model.reg_model.coef_)

print(lasso_model2.reg_model.coef_ - lasso_model.reg_model.coef_)
