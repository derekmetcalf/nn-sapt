import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def scatter_plot_from_file(filename, plot_title):
    file = open(filename,"r")
    lines = file.readlines()
    total_mae = float(lines[1].split(",")[1])
    total = []
    total_pred = []
    total_std_dev = []

    for i in range(len(lines)):

        if i>6:
            total.append(float(lines[i].split(",")[1].replace("[","").replace("]","")))
            total_pred.append(float(lines[i].split(",")[2].replace("[","").replace("]","")))
            total_std_dev.append(float(lines[i].split(",")[3].replace("[","").replace("]","")))

    file.close()
    
    total = np.asarray(total)
    total_pred = np.asarray(total_pred)
    total_std_dev = np.asarray(total_std_dev)
 
    fig = plt.figure(figsize=(20,20))
    ax0 = fig.add_subplot(111)
    #im = ax0.scatter(total, total_pred, alpha=0.5, c="xkcd:black")
    im = ax0.scatter(total, total_pred, alpha=0.8)#, c=total_std_dev, cmap="viridis")
    
    #fig.colorbar(im)

    lims = [np.min([ax0.get_xlim(), ax0.get_ylim()]),
            np.max([ax0.get_xlim(), ax0.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax0.set_aspect('equal')
    ax0.set_xlim(lims)
    ax0.set_ylim(lims)
    ax0.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax0.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax0.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax0.text(0.1,0.9,"MAE = %.2f kcal/mol"%(total_mae),
                fontsize=14,transform=ax0.transAxes)
    ax0.grid(color='k',linestyle=':',alpha=0.4)
    ax0.set_title(plot_title)
    ax0.set_xlabel("SAPT0 Interaction Energy (kcal/mol)")
    ax0.set_ylabel("NN Predicted Interaction Energy (kcal/mol)")
    #plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
    #fig.text(0.5, 0.1, "SAPT0 Interaction Energy (kcal/mol)", ha='center')
    #fig.text(0.1, 0.5, "NN Predicted Interaction Energy (kcal/mol)", 
    #            va="center", rotation="vertical")
    #plt.suptitle(plot_title), x=0.5, y=0.9, ha="center")
    #plt.subplots_adjust(wspace=0.4, hspace=0.0)
    plt.show()
    return

scatter_plot_from_file("../results/paper_stuff/bare-nma-aniline-rand_results_model_UNUSED.csv","")
#scatter_plot_from_file("./test_results/0.0125/neutral-SSI_0.0125-spike_100-100-75NMe-acetamide_Don--Benzimidazole-PDB-xyzcoords-nrgs-dmv2-format.csv",
#                    "")
#plot_sapt_from_file("./NMA-Aniline-0.1-crystal-pert.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Unlabeled Perturbed Dimers")
#plot_sapt_from_file("./artif_and_labeled_pert_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Labeled Perturbed Dimers")
#plot_sapt_from_file("./symfun_experiment_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data, Labeled Perturbed Dimers, and 'Intermolecular' Symmetry Functions")
