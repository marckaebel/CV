import data_generation
import plotting_helpers
import numpy as np
import matplotlib.pyplot as plt



###################################### Colors ###################################################



colors = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]
black_to_orange = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#dc2f02", "#e85d04", "#f48c06", "#faa307", "#ffba08"]
blues = ["#03045e","#0077b6","#00b4d8","#90e0ef","#caf0f8"]
blues_2 = ["#051923","#003554","#006494","#0582ca","#00a6fb"]
blues_2_reverse = ["#00a6fb","#0582ca","#006494","#003554","#051923"]
pink_to_blue = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
greens = ["#004b23","#006400","#007200","#008000","#38b000","#70e000","#9ef01a","#ccff33"]
bright_colors = ["#ffbe0b","#fb5607","#ff006e","#8338ec","#3a86ff"]
warm_earth = ["#6f1d1b","#bb9457","#432818","#99582a","#ffe6a7"]
monochrome = ["#f8f9fa","#e9ecef","#dee2e6","#ced4da","#adb5bd","#6c757d","#495057","#343a40","#212529"]





#################################################################################################
###################################### Execution ################################################

# Uncommenent the desired code block here to create a plot



'''
Scattering normal distribution, 2D Normal Distribution
'''

# default_width = 5.78851 # in inches
# default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

# def get_correlated_dataset(n, dependency, mu, scale):
#     '''
#     Creates data for a 2D random distribution
#     '''
#     latent = np.random.randn(n, 2)
#     dependent = latent.dot(dependency)
#     scaled = dependent * scale
#     scaled_with_offset = scaled + mu
#     # return x and y of the new, correlated dataset
#     return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

# Sigma = [[0.9, -0.4],
#         [0.1, -0.6]]

# mu = 2, 4
# scale = 3, 5
# x, y = get_correlated_dataset(800, Sigma, mu, scale)


# x_samples = np.linspace(0, 1, 500)
# kernel = data_generation.Matern(length_scale=0.2)
# n_samples=3
# y_mean, y_conf, y_samples = data_generation.gaussian_regression_1D(x_samples, kernel, n_samples=n_samples)

# fig, axs = plt.subplots(1,3, figsize=(default_width, default_width*default_ratio/2))
# axs[0].scatter(y[:50],np.zeros(50),s=0.5, color = bright_colors[0])
# axs[1].scatter(x, y, s=0.5, color = bright_colors[0])
# # x= np.zeros(800)
# for i,y_s in enumerate(y_samples):
#     axs[2].plot(x_samples, y_s, color = bright_colors[i])

# plt.setp([(a.get_yticklabels(),a.get_xticklabels()) for a in axs], visible=False)
# plt.setp([a.tick_params(left = False, bottom = False) for a in axs])
# [spine.set_visible(False) for a in axs for spine in a.spines.values()]  
# [a.set_box_aspect(1) for a in axs]
# plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
# plt.tight_layout()
# # plt.savefig('figures/random_variable_random_vector_random_process.pgf')
# plt.show()

'''
Optimization: f* and x* plot
'''
# x = np.linspace(-4.5,0,1000)
# y = np.sin(x**2+1-1/(x+5))+0.4*x+0.02*x**4-0.1*np.exp(x)
# fig = plotting_helpers.plot_functions_plain(x, colors, y)
# # plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
# plt.plot([-2, -2],[-1.5 ,6], color= 'black', linewidth=0.5,linestyle = '--',markevery=[-1], marker='|', markersize=6)
# plt.plot([-2, -4],[-1.5 ,-1.5], color = 'black', linewidth=0.5,linestyle = '--',markevery=[-1], marker='_', markersize=6)
# plt.text(-1.97, 6.85, r"$x^*$", horizontalalignment='center',
#      verticalalignment='center')
# plt.text(-4.2, -1.5, r"$f^*$", horizontalalignment='center',
#      verticalalignment='center')
# plt.savefig('figures/optimization_intro.pgf')

# plt.show()



'''
Plotting kernels with negative numbers too
'''
# x = np.linspace(0, 6, 1000)[1:]
# # y_values = (data_generation.matern(x, nu=1/3), data_generation.matern(x,nu=2), data_generation.sq_exponential(x))
# y_values = (data_generation.sq_exponential(x), data_generation.matern(x, nu=1/3) )
# y_values = [ np.concatenate((np.flip(a), a)) for a in y_values]
# # legend_data = (r'Matérn with $\nu = 0.3$', r'Matérn with $\nu = 2$', r'Squared exponential')
# legend_data = (r"$K_{M^{1/3}}$", r"$K_{M^{2}}$", r"$K_{SE}$")
# axis_text=(r"$|x-x'|$",r"$K(x,x')$")
# axis_text=(r"$x$", r"$p(x)$")
# x = np.linspace(-6, 6, 1998)
# fig = plotting_helpers.plot_kernel_functions(x, axis_text, colors, *y_values)# , legend_data=legend_data)
# plt.subplots_adjust(left = 0, top = 0.95, right = 0.95, bottom = 0, hspace = 0.5, wspace = 0.5)

# plt.savefig('figures/tail_distribution_kernels.pgf')
# plt.show()

'''
Plot Matérn kernel and squared exponential kernel
'''
# x = np.linspace(-4, 4, 1000)[1:]
# y_values = (data_generation.matern(np.abs(x), nu=1/3), data_generation.matern(np.abs(x),nu=1), data_generation.sq_exponential(np.abs(x)))
# # legend_data = (r'Matérn with $\nu = 0.3$', r'Matérn with $\nu = 2$', r'Squared exponential')
# legend_data = (r"$C_{1/3}$", r"$C_{1}$", r"$C_{SE}$")
# axis_text=(r"$d$",r"$C(d)$")
# fig = plotting_helpers.plot_kernel_functions(x, axis_text, colors, *y_values, legend_data=legend_data)
# plt.savefig('figures/covariance_kernels_plot.pgf')
# plt.show()



'''
Plot Matérn kernels for different nu
'''
x = np.linspace(-4, 4, 1000)[1:]
y_values = (data_generation.matern(np.abs(x), nu=1/3), 
            data_generation.sq_exponential(np.abs(x)), 
            data_generation.exponential(np.abs(x)),
            )
legend_data = (r"$\nu = 1/3$",
                "squared exponential",
                "exponential"
                )
axis_text=(r"$d$",r"$K(x,x')$")
fig = plotting_helpers.plot_kernel_functions(x, axis_text, colors, *y_values, legend_data=legend_data)
plt.show()




'''
Plot different kernels using the sklearn toolkit
'''
# x = np.linspace(0, 4, 100)
# kernels = [data_generation.RBF(), data_generation.Matern(nu=1.5)]
# y=[kernel(np.reshape(x,(-1,1)))[0] for kernel in kernels] 
# legend_data = ('$K_{SE}$', '$K_{M^{1}}$', '$K_{SE}$','$K_{White}$')
# axis_text=["d","K(x,x\')"]
# plotting_helpers.plot_kernel_functions(x, axis_text, colors, *y, legend_data=legend_data)



'''
Plot normal distributions
'''
# x = np.linspace(0, 3, 100)
# y = [data_generation.gaussian_pdf(x,1,0.4), data_generation.gaussian_pdf(x,1.5,0.2)] 
# # legend_data = ('Gaussian', 'also Gaussian')
# axis_text=[r"$x$",r"$p(x)$"]
# fill=[True, True]
# linestyle=['solid','dashed']
# fig = plotting_helpers.plot_kernel_functions(x, axis_text, colors, *y, fill=fill, linestyle=linestyle)
# plt.savefig('figures/normal_distribution_plot.pgf')
# plt.show()



'''
Minimal plot of n samples drawn from Gaussian process with specified covariance kernel
Choose kernel from 
- RBF
- Matern
- WhiteKernel
- ExpSineSquared
- ConstantKernel
- product or linear combination of any of the above
'''
# x = np.linspace(0, 1, 500)
# kernel = data_generation.RBF(length_scale=0.1)
# n_samples=4
# y_mean, y_conf, y_samples = data_generation.gaussian_regression_1D(x, kernel, n_samples=n_samples)
# fig = plotting_helpers.plot_functions_plain(x, bright_colors, *y_samples) #, title=r"$\nu=1/2$")
# # plt.savefig('figures/samples_drawn_ConstantKernel.pgf')
# plt.show()




'''
Minimal plot of a function and scattered points sampled randomly
'''
# x = np.linspace(0, 3, 100)
# f = lambda x: np.sin((x-2.5)**2)
# y = f(x)
# points = data_generation.sample_points(f,x,20)
# plotting_helpers.plot_functions_plain(x, blues[1:], y, points = points)



'''
Minimal plot of n samples drawn from Gaussian process with specified covariance kernel
Choose kernel from 
- RBF
- Matern
- WhiteKernel
- ExpSineSquared
- ConstantKernel
- product or linear combination of any of the above
Multiple plots next to each other
'''
# x = np.linspace(0, 1, 500)
# kernels = [
#     data_generation.Matern(length_scale=0.1, nu=0.5),
#     data_generation.Matern(length_scale=0.1, nu=1.5),
#     data_generation.Matern(length_scale=0.1, nu=2.5),
#     # data_generation.Matern(length_scale=0.1, nu=3.5)
#     ]
# title = [
#     r"$\nu=1/2$",
#     r"$\nu=3/2$",
#     r"$\nu=5/2$",
#     # r"$\nu=7/2$"
#     ]
# n_samples=3
# y_mean = []
# y_conf = []
# y_samples = [] #np.array([]) 
# for kernel in kernels:
#     y_m, y_c, y_s = data_generation.gaussian_regression_1D(x, kernel, n_samples=n_samples)
#     y_mean.append(y_m)
#     y_conf.append(y_c)
#     y_samples.append(y_s)
# # plot
# fig = plotting_helpers.plot_multiple_functions_plain(x, bright_colors, *y_samples, title=title)
# plt.savefig('figures/samples_drawn_different_Matern.pgf')
# plt.show()



'''
TBD:
Minimal plot of a function and scattered points from Newtons or CG method
Want to show: that minimization can get stuck in a local minimum if unlucky
'''
# from scipy import optimize
# x = np.linspace(0, 3, 100)
# f = lambda x: np.sin((x-2.5)**2)
# y = f(x)

# vec_res = optimize.newton(f, x, fprime=fder, args=(a, ), maxiter=200)
# points = data_generation.sample_points(f,x,20)
# plotting_helpers.plot_functions_plain(x, blues[1:], y, points = points)



'''
Wide plot 1D Gaussian Regression prior/posterior
'''
# x = np.linspace(0, 5, 100) 
# f = lambda x: np.sin(x**2) #would it be better to take a sample from the kernel instead of a predefined function?
# n_eval=6
# n_samples=4
# eval_points = data_generation.sample_points(f,x,n_eval)
# kernel = data_generation.RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
# y_mean, y_conf, y_samples = data_generation.gaussian_regression_1D(x, kernel, n_samples=n_samples, eval_points=eval_points)
# plotting_helpers.plot_functions_wide(x, bright_colors, y_mean, y_conf, eval_points, *y_samples)



'''
Plot a random field
'''
# size=1024
# spectrum=data_generation.spectrum_power_series
# # spectrum_coeff=[600] # Cosine spectrum: length scale
# spectrum_coeff=[4] # Power series spectrum: power coeffient
# # spectrum_coeff = [20,2,1000,100] # Matern spectrum: nu, dimension n, sigma, length scale
# # spectrum_coeff=[1000] # Square exponential: length scale

# field = data_generation.random_field(size,spectrum,spectrum_coeff=spectrum_coeff)
# # field = data_generation.random_field_anisotropic() # this one does one anisotropic example without customization
# fig = plotting_helpers.plot_field(field)
# plt.savefig('figures/random_field.pgf')
# plt.show()






'''
Plot 2D Gaussian regression from sample points
'''
# generating the random field
# size = 512
# spectrum = data_generation.spectrum_matern
# spectrum_coeff = [3/2,2,1000,200] # Matern spectrum: nu, dimension n, sigma, length scale
# field = data_generation.random_field(size,spectrum,spectrum_coeff=spectrum_coeff)
# # sampling from the random field
# n_samples = 100
# points = data_generation.sample_points_2D(n_samples, field,size)
# # gaussian process regression
# x = np.arange(size)
# X = np.meshgrid(x, x, indexing='ij')
# kernel = data_generation.Matern(length_scale=200)
# y_mean, y_conf = data_generation.gaussian_regression_2D(X, points, kernel)
# # plotting
# plotting_helpers.plot_gaussian_regression_2D(points, y_mean.reshape(size,size), field)






'''
Showing the normal distribution of samples evaluated at a t in T

'''

# default_width = 5.78851 # in inches
# default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

# x = np.linspace(0, 1, 500)
# kernels = [
#     data_generation.RBF(length_scale=0.1),
#     ]
# n_samples=80
# y_mean = []
# y_conf = []
# y_samples = [] #np.array([]) 
# for kernel in kernels:
#     y_m, y_c, y_s = data_generation.gaussian_regression_1D(x, kernel, n_samples=n_samples)
#     y_mean.append(y_m)
#     y_conf.append(y_c)
#     y_samples.append(y_s)

# eval = np.array(y_samples)[0,0:n_samples,400]
# t = np.linspace(-4, 4, 100)
# y_t = data_generation.gaussian_pdf(t,0,.7)
# fig, axs = plt.subplots(1,2, figsize=(default_width, default_width*default_ratio*0.6))
# for j,y in enumerate(y_samples[0]):
#      axs[0].plot(x,y, color=bright_colors[j % len(bright_colors)], linewidth=0.7)

# axs[1].scatter(eval,np.zeros(n_samples),s=1.5,  c=[bright_colors[i % len(bright_colors)] for i in range(n_samples)])
# axs[1].plot(t,y_t, c=colors[0])
# axs[0].set_xlim(xmin=0, xmax=1)
# axs[0].set_ylim(ymin=-4, ymax=4)
# axs[0].set_xlabel(r"$x$")
# axs[0].set_ylabel(r"$f(x)$")
# axs[1].set_xlim(xmin=-4, xmax=4)
# axs[1].set_xlabel(r"$f(x^*)$")
# axs[1].set_ylabel(r"$p_{f(x^*)}$")
# axs[0].plot([0.8, 0.8],[-10 ,10], color= 'black', linewidth=0.8,linestyle = '--',markevery=[-1], marker='|', markersize=6)
# axs[0].text(0.8, -4.5, r"$x^*$", horizontalalignment='center',    verticalalignment='center')


# plt.setp([(a.get_yticklabels(),a.get_xticklabels()) for a in axs], visible=False)
# plt.setp([a.tick_params(left = False, bottom = False) for a in axs])
# # [spine.set_visible(False) for a in axs for spine in a.spines.values()]  
# # [a.set_box_aspect(1) for a in axs]
# plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
# plt.tight_layout()
# # plt.savefig('figures/gaussian_process_evaluated.pgf')
# plt.show()



#################################################################################################
#################################################################################################
