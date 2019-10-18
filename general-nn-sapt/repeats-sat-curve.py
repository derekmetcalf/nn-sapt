import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

def scatter_plot_from_file():
    matplotlib.rcParams['axes.linewidth'] = 12.5
   
    comp_frac = [25,50,100,200] #num configs per dimer
    tot_mae = [0.8381, 0.6958, 0.6830, 0.6760]
    elst_mae = [0.8519, 0.7086, 0.7069, 0.7210]
    exch_mae = [0.6751, 0.5677, 0.5006, 0.5485]
    ind_mae = [0.2017, 0.1861, 0.1904, 0.2052]
    disp_mae = [0.2334, 0.1809, 0.1913, 0.2109]
 
    fig = plt.figure(figsize=(25,20))
    ax0 = fig.add_subplot(111)
    im0 = ax0.scatter(comp_frac, tot_mae, alpha=0.8, s=600, c="xkcd:black", zorder=10)
    line0, = ax0.plot(comp_frac, tot_mae, linewidth=10, c="xkcd:black", zorder=10)
    line0.set_label("Total")
    im1 = ax0.scatter(comp_frac, elst_mae, alpha=0.8, s=600, c="xkcd:red")
    line1, = ax0.plot(comp_frac, elst_mae,  linewidth=10, linestyle="dashed", c="xkcd:red")
    line1.set_label("Electrostatics")
    im2 = ax0.scatter(comp_frac, exch_mae, alpha=0.8, s=600, c="xkcd:green")
    line2, = ax0.plot(comp_frac, exch_mae,  linewidth=10, linestyle="dashed", c="xkcd:green")
    line2.set_label("Exchange")
    im3 = ax0.scatter(comp_frac, ind_mae, alpha=0.8, s=600, c="xkcd:blue")
    line3, = ax0.plot(comp_frac, ind_mae,  linewidth=10, linestyle="dashed", c="xkcd:blue")
    line3.set_label("Induction")
    im4 = ax0.scatter(comp_frac, disp_mae, alpha=0.8, s=600, c="xkcd:orange")
    line4, = ax0.plot(comp_frac, disp_mae,  linewidth=10, linestyle="dashed", c="xkcd:orange")
    line4.set_label("Dispersion")

    #ax0.set_aspect('square')
    ax0.set_xlim([0, 250])
    ax0.set_ylim([0, 1.2])
    
    ax0.set_xlabel(r"Training configurations per NMA/X dimer", fontsize=55)
    ax0.set_ylabel("Validation MAE (kcal mol$^{-1}$)", fontsize=55)
    ax0.tick_params(axis="x", direction="in", labelsize=50, pad=25, length=25, width=12.5)
    ax0.tick_params(axis="y", direction="in", labelsize=50, pad=25, length=25, width=12.5)
    #ax0.xaxis.set_major_locator(plt.MaxNLocator(4))
    #ax0.yaxis.set_major_locator(plt.MaxNLocator(4))

    ax0.legend(fontsize=50, frameon=False)
    plt.tight_layout(pad=0.4)
    plt.savefig("multitarget_sat_curve.png", dpi=300)

    #ax0.spines['right'].set_visible(False)
    #ax0.spines['top'].set_visible(False)
    #plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
    #fig.text(0.5, 0.1, "SAPT0 Interaction Energy (kcal/mol)", ha='center')
    #fig.text(0.1, 0.5, "NN Predicted Interaction Energy (kcal/mol)", 
    #            va="center", rotation="vertical")
    #plt.suptitle(plot_title), x=0.5, y=0.9, ha="center")
    #plt.subplots_adjust(wspace=0.4, hspace=0.0)
    #plt.show()

    return

scatter_plot_from_file()
#scatter_plot_from_file("./step_3_testing_NMA-Aniline-crystallographic.csv",
#plot_sapt_from_file("./NMA-Aniline-0.1-crystal-pert.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Unlabeled Perturbed Dimers")
#plot_sapt_from_file("./artif_and_labeled_pert_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Labeled Perturbed Dimers")
#plot_sapt_from_file("./symfun_experiment_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data, Labeled Perturbed Dimers, and 'Intermolecular' Symmetry Functions")
