from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import data_generation
import plotting_helpers


import matplotlib.pyplot as plt
from matplotlib import gridspec


def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x = np.linspace(-2, 10, 1000).reshape(-1, 1)
y = target(x)


optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=27)

acq_function = UtilityFunction(kind="ucb")#, kappa=5)
optimizer.maximize(init_points=1, n_iter=0, acquisition_function = acq_function)


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma






optimizer.maximize(init_points=0, n_iter=10, acquisition_function=acq_function)

x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
y_obs = np.array([res["target"] for res in optimizer.res])
eval_points= [x_obs.flatten(), y_obs.flatten()]

y_mean, y_conf = posterior(optimizer, x_obs, y_obs, x)

utility_function = UtilityFunction(kind="ucb")#, kappa=5, xi=0)
utility = utility_function.utility(x, optimizer._gp, 0)

x = np.linspace(-2, 10, 1000)

plotting_helpers.plot_bayesopt_wide(x, y, y_mean, 1.9600*y_conf, 0.5*utility-1.5, eval_points)
plt.show()






# x = np.linspace(0, 5, 100) 
# f = lambda x: np.sin(x**2) #would it be better to take a sample from the kernel instead of a predefined function?
# n_eval=6
# n_samples=4
# eval_points = data_generation.sample_points(f,x,n_eval)
# kernel = data_generation.RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
# y_mean, y_conf, y_samples = data_generation.gaussian_regression_1D(x, kernel, n_samples=n_samples, eval_points=eval_points)

# plotting_helpers.plot_functions_wide(x, bright_colors, y_mean, y_conf, eval_points, *y_samples)



# import matplotlib.pyplot as plt

# import matplotlib.patches as patches

# colors = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]

# fig, ax = plt.subplots(1, 1, figsize=(5, 2),
#                         dpi=80, facecolor='w', edgecolor='k')

# for sp in ax.spines.values():
#     sp.set_visible(False)

# plt.xticks([])
# plt.yticks([])

# bars = plt.bar(range(len(colors)), [1]*len(colors), color=colors)
# plt.show()