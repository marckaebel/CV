import matplotlib.pyplot as plt
import numpy as np
# import data_generation # can go when execution is moved to make_plots
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.axes_grid1 import ImageGrid


default_width = 5.78851 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean
plt.rcParams.update(
{
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "pgf.rcfonts": False, 
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       10,         # control font sizes of different elements
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pgf.preamble": "\n".join([              # specify additional preamble calls for LaTeX's run
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{siunitx}",
        r"\usepackage{amssymb}",
        r"\usepackage{amsmath}"

    ]),
    "figure.figsize" : [default_width, default_width * default_ratio]
})




# fig.legend(loc= 'outside upper center', ncols=4, borderaxespad=0., frameon=False)
# ax.set_xlim(xmin = np.min(x), xmax = np.max(x))
# plt.subplots_adjust(left = 0, top = 0.8, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
# # fig.set_tight_layout({'pad': 0 })
# plt.savefig('figures\sausage_plot_posterior_matern.pgf')#, bbox_inches='tight')#, pad_inches=0)
# plt.show()


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif", # or mathpazo for style
#     "font.size": "10"
# })




'''
This file has different functions to plot either (x, y) or (x, y, z) data 
'''

#################################################################################################
###################################### Functions ################################################



def plot_kernel_functions(x, axis_text, colors, *y_values, **kwargs):
    '''
    Takes x array, legend data, color palette and y values
    Prints out a plot with x and y axis arrows and axis labels
    '''
    fill = kwargs.get('fill')
    linestyle = kwargs.get('linestyle')
    legend_data = kwargs.get('legend_data')
    fig = plt.figure(figsize=(default_width*0.75, default_width*default_ratio*0.5))
    ax = fig.add_subplot(axes_class=AxesZero)

    for i,y in enumerate(y_values):#range(len(y_values)):
        if linestyle is not None:
            if legend_data is not None:
                ax.plot(x,y, label=legend_data[i], color=colors[i], linestyle=linestyle[i])
            else: 
                ax.plot(x,y, color=colors[i], linestyle=linestyle[i])
        else:
            if legend_data is not None:
                ax.plot(x,y, label=legend_data[i], color=colors[i])
            else:
                ax.plot(x,y, color=colors[i])
        if fill is not None and fill[i] == True: ax.fill_between(x, y, alpha=0.1, color=colors[i])
    if legend_data is not None:
        ax.legend(frameon=False, loc="best")   

    for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin=-4)
    ax.set_ylim(ymin=0)

    ax.text(np.max(x), 0.05*np.max(y_values), axis_text[0])
    ax.text(0.05*np.max(x), np.max(y_values), axis_text[1])
    # plt.tight_layout()
    return fig




def plot_functions_plain(x, colors, *y_values, **kwargs):
    '''
    Takes x array, color palette, y values and possible coordinates at which to place dots
    Prints out a plain plot
    '''
    points = kwargs.get('points')
    title = kwargs.get('title')

    fig, ax = plt.subplots(figsize=(0.6*default_width,0.6*default_width*default_ratio*0.6))
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    # ax.set_box_aspect(1)
    for i,y in enumerate(y_values):
        ax.plot(x,y, color=colors[i])
    if points is not None:
        ax.scatter(points[0], points[1], color="black", zorder=10,s=15)
    if title is not None:
        ax.set_title(title)
        plt.subplots_adjust(left = 0, top = 0.88, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    else: 
        plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    return fig
    # plt.show()

def plot_multiple_functions_plain(x, colors, *y_values, **kwargs):
    '''
    Takes x array, color palette, y values and titles
    Prints out plain plots
    '''
    title = kwargs.get('title')
    linewidth= kwargs.get('linewidth')
    k=np.shape(y_values)[0]
    fig = plt.figure(figsize=(default_width, default_width/3))
    grid = ImageGrid(
    fig, 111, nrows_ncols=(1, k), axes_pad=0.1, share_all=True, aspect=False)
    for i,y_val in enumerate(y_values):
        for j,y in enumerate(y_val):
            grid[i].plot(x,y, color=colors[j % len(colors)], linewidth=linewidth)
        if title is not None:
            grid[i].set_title(title[i])
            plt.subplots_adjust(left = 0, top = 0.8, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
        else: 
            plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    
    # plt.setp([(a.set_box_aspect(1)) for a in grid])
    plt.setp([(a.get_yticklabels(),a.get_xticklabels()) for a in grid], visible=False)
    plt.setp([a.tick_params(left = False, bottom = False) for a in grid])
    [spine.set_visible(False) for a in grid for spine in a.spines.values()]    
    return fig






def plot_field(y):
    '''
    Plots a random field with a colormap
    '''
    fig, ax = plt.subplots(figsize=(0.33*default_width, 0.33*default_width))
    ax.imshow(y, cmap="viridis", aspect="equal")
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    [spine.set_visible(False) for spine in ax.spines.values()]
    plt.tight_layout()
    plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    return fig



def plot_functions_wide(x, colors, y_mean, y_conf=None, eval_points = None, *y_samples):
    '''
    This function is supposed to plot
    - sample functions
    - mean function
    - confidence intervals
    - markers of function evaluation
    for both prior and posteriors.
    '''
    fig, ax = plt.subplots(figsize=(5.78851, 1))
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if eval_points[0].any():
        ax.scatter(eval_points[0], eval_points[1], color="black", zorder=10, label="Observations",s=15)
    if y_samples:
        for i,y in enumerate(y_samples):
            if i==0: ax.plot(x,y, color=colors[i], label="Samples")
            else: ax.plot(x,y, color=colors[i])
    
    ax.plot(x, y_mean, color="#264653", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_conf,
        y_mean + y_conf, 
        alpha=0.1,
        color="#264653",
        label=r"$95\%$ credible interval",
    )
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .1), loc='center', ncols=4, borderaxespad=0., frameon=False) #maybe mode="expand",
    fig.legend(loc= 'outside upper center', ncols=4, borderaxespad=0., frameon=False)
    ax.set_xlim(xmin = np.min(x), xmax = np.max(x))
    plt.subplots_adjust(left = 0, top = 0.8, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    # fig.set_tight_layout({'pad': 0 })
    # plt.savefig('figures\sausage_plot_posterior_matern.pgf')#, bbox_inches='tight')#, pad_inches=0)
    plt.show()



def plot_gaussian_regression_2D(points, interpolation, field,):
    '''
    Creates 3 plots: a scatter plot of samples, interpolation and a random field 
    Also includes a correctly sized color map
    '''
    fig = plt.figure(figsize=(default_width, default_width/3))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 3), axes_pad=0.1, share_all=True,
        cbar_location="right", cbar_mode="single", cbar_size="10%", cbar_pad=0.1)

    im = grid[0].scatter(points[1,:], points[0,:], c = points[2,:],s=15, cmap = "viridis", clip_on = False)
    im = grid[1].imshow(interpolation, cmap="viridis", aspect="equal", origin='lower')
    im = grid[2].imshow(field, cmap="viridis", aspect="equal", origin='lower')    

    plt.setp([(a.get_yticklabels(),a.get_xticklabels()) for a in grid], visible=False)
    plt.setp([a.tick_params(left = False, bottom = False) for a in grid])
    [spine.set_visible(False) for a in grid for spine in a.spines.values()]    
    cb = grid[2].cax.colorbar(im)
    # cb.outline.set_visible(False)
    cb.set_ticks([])
    plt.subplots_adjust(left = 0, top = 0.95, right = 1, bottom = 0.05, hspace = 0.5, wspace = 0.5)
    # fig.set_tight_layout({'pad': 0 })
    plt.savefig('figures/random_field_regressiom.pgf')#, bbox_inches='tight')#, pad_inches=0)

    plt.show()



#################################################################################################
    
def plot_bayesopt_wide(x, target, y_mean, y_conf, utility_func, eval_points = None):
    '''
    This function is supposed to plot
    - target function
    - prediction function
    - confidence intervals
    - markers of function evaluation
    - the utility function
    '''
    fig, ax = plt.subplots(figsize=(5.78851, 2))
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if eval_points[0].any():
        ax.scatter(eval_points[0], eval_points[1], color="black", zorder=10, label="Samples",s=15)   
    ax.plot(x, target, color="#264653", label="Target", linestyle="--", linewidth=1) 
    ax.plot(x, y_mean, color="#2a9d8f", label="Prediction")

    ax.fill_between(
        x,
        y_mean - y_conf,
        y_mean + y_conf, 
        alpha=0.1,
        color="#2a9d8f",
        label=r"$95\%$ credible interval",
    )

    ax.fill_between(
        x,
        utility_func,
        # 0,
        -3, 
        alpha=0.1,
        color="#f4a261",
    )
    ax.plot(x, utility_func, color="#f4a261", label="Utility")
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .1), loc='center', ncols=4, borderaxespad=0., frameon=False) #maybe mode="expand",
    fig.legend(loc= 'outside upper center', ncols=5, borderaxespad=0, frameon=False, labelspacing=0.1)
    ax.set_xlim(xmin = np.min(x)-0.1, xmax = np.max(x)+0.1)
    ax.set_ylim(ymin = -1.5)
    plt.subplots_adjust(left = 0, top = 0.96, right = 1, bottom = 0, hspace = 0.5, wspace = 0.5)
    # fig.set_tight_layout({'pad': 0 })
    # plt.savefig('figures/bayes_opt_ucb.pgf')#, bbox_inches='tight')#, pad_inches=0)
    plt.show()
