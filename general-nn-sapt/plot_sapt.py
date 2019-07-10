import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_sapt_from_file(filename, plot_title):
    file = open(filename,"r")
    lines = file.readlines()
    total_mae = float(lines[1].split(",")[1])
    elst_mae = float(lines[2].split(",")[1])
    exch_mae = float(lines[3].split(",")[1])
    ind_mae = float(lines[4].split(",")[1])
    disp_mae = float(lines[5].split(",")[1])
    total = []
    total_pred = []
    elec = []
    elec_pred = []
    exch = []
    exch_pred = []
    ind = []
    ind_pred = []
    disp = []
    disp_pred = []

    for i in range(len(lines)):

        if i>6:
            total.append(float(lines[i].split(",")[1].replace("[","").replace("]","")))
            total_pred.append(float(lines[i].split(",")[2].replace("[","").replace("]","")))
            elec.append(float(lines[i].split(",")[4].replace("[","").replace("]","")))
            elec_pred.append(float(lines[i].split(",")[5].replace("[","").replace("]","")))
            exch.append(float(lines[i].split(",")[7].replace("[","").replace("]","")))
            exch_pred.append(float(lines[i].split(",")[8].replace("[","").replace("]","")))
            ind.append(float(lines[i].split(",")[10].replace("[","").replace("]","")))
            ind_pred.append(float(lines[i].split(",")[11].replace("[","").replace("]","")))
            disp.append(float(lines[i].split(",")[13].replace("[","").replace("]","")))
            disp_pred.append(float(lines[i].split(",")[14].replace("[","").replace("]","")))

    file.close()
    
    total = np.asarray(total)
    total_pred = np.asarray(total_pred)
    elec = np.asarray(elec)
    elec_pred = np.asarray(elec_pred)
    exch = np.asarray(exch)
    exch_pred = np.asarray(exch_pred)
    ind = np.asarray(ind)
    ind_pred = np.asarray(ind_pred)
    disp = np.asarray(disp)
    disp_pred = np.asarray(disp_pred)
    
    fig = plt.figure(figsize=(20,10))
    ax0 = plt.subplot2grid((2,4), (0,0), rowspan=2, colspan=2)
    ax0.scatter(total, total_pred, c="xkcd:black", alpha=0.2)
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
    ax0.set_title("Total Interaction Energy")
    
    ax1 = plt.subplot2grid((2,4), (0,2))
    ax1.scatter(elec, elec_pred, c="xkcd:red", s=7)
    lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]),
            np.max([ax1.get_xlim(), ax1.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax1.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax1.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax1.text(0.1,0.9,"MAE = %.2f kcal/mol"%(elst_mae),
                fontsize=10,transform=ax1.transAxes)
    ax1.grid(color='k',linestyle=':',alpha=0.4)
    ax1.set_title("Electrostatics")

    ax2 = plt.subplot2grid((2,4), (0,3))
    ax2.scatter(exch, exch_pred, c="xkcd:green", s=7)
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax2.set_aspect('equal')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax2.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax2.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax2.text(0.1,0.9,"MAE = %.2f kcal/mol"%(exch_mae),
                fontsize=10,transform=ax2.transAxes)
    ax2.grid(color='k',linestyle=':',alpha=0.4)
    ax2.set_title("Exchange")

    ax3 = plt.subplot2grid((2,4), (1,2))
    ax3.scatter(ind, ind_pred, c="xkcd:blue", s=7)
    lims = [np.min([ax3.get_xlim(), ax3.get_ylim()]),
            np.max([ax3.get_xlim(), ax3.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax3.set_aspect('equal')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax3.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax3.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax3.text(0.1,0.9,"MAE = %.2f kcal/mol"%(ind_mae),
                fontsize=10,transform=ax3.transAxes)
    ax3.grid(color='k',linestyle=':',alpha=0.4)
    ax3.set_title("Induction")

    ax4 = plt.subplot2grid((2,4), (1,3))
    ax4.scatter(disp, disp_pred, c="xkcd:orange", s=7)
    lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]),
            np.max([ax4.get_xlim(), ax4.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax4.set_aspect('equal')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax4.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax4.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax4.text(0.1,0.9,"MAE = %.2f kcal/mol"%(disp_mae),
                fontsize=10,transform=ax4.transAxes)
    ax4.grid(color='k',linestyle=':',alpha=0.4)
    ax4.set_title("Dispersion")

    #plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
    fig.text(0.5, 0.1, "SAPT0 Interaction Energy (kcal/mol)", ha='center')
    fig.text(0.1, 0.5, "NN Predicted Interaction Energy (kcal/mol)", 
                va="center", rotation="vertical")
    plt.suptitle(plot_title, x=0.5, y=0.9, ha="center")
    plt.subplots_adjust(wspace=0.4, hspace=0.0)
    plt.show()
    return

def plot_sapt_from_folder(foldername, plot_title):
    filename = []
    for r,d,f in os.walk(foldername):
        for name in f:
            filename.append(os.path.join(foldername,name))
    total = []
    total_pred = []
    elec = []
    elec_pred = []
    exch = []
    exch_pred = []
    ind = []
    ind_pred = []
    disp = []
    disp_pred = []
    for name in filename:
        file = open(name,"r")
        lines = file.readlines()
        total_mae = float(lines[1].split(",")[1])
        elst_mae = float(lines[2].split(",")[1])
        exch_mae = float(lines[3].split(",")[1])
        ind_mae = float(lines[4].split(",")[1])
        disp_mae = float(lines[5].split(",")[1])


        for i in range(len(lines)):

            if i>6 and i<40:
                total.append(float(lines[i].split(",")[1].replace("[","").replace("]","")))
                total_pred.append(float(lines[i].split(",")[2].replace("[","").replace("]","")))
                elec.append(float(lines[i].split(",")[4].replace("[","").replace("]","")))
                elec_pred.append(float(lines[i].split(",")[5].replace("[","").replace("]","")))
                exch.append(float(lines[i].split(",")[7].replace("[","").replace("]","")))
                exch_pred.append(float(lines[i].split(",")[8].replace("[","").replace("]","")))
                ind.append(float(lines[i].split(",")[10].replace("[","").replace("]","")))
                ind_pred.append(float(lines[i].split(",")[11].replace("[","").replace("]","")))
                disp.append(float(lines[i].split(",")[13].replace("[","").replace("]","")))
                disp_pred.append(float(lines[i].split(",")[14].replace("[","").replace("]","")))

        file.close()
    
    total = np.asarray(total)
    total_pred = np.asarray(total_pred)
    elec = np.asarray(elec)
    elec_pred = np.asarray(elec_pred)
    exch = np.asarray(exch)
    exch_pred = np.asarray(exch_pred)
    ind = np.asarray(ind)
    ind_pred = np.asarray(ind_pred)
    disp = np.asarray(disp)
    disp_pred = np.asarray(disp_pred)
    
    fig = plt.figure(figsize=(20,10))
    ax0 = plt.subplot2grid((2,4), (0,0), rowspan=2, colspan=2)
    ax0.scatter(total, total_pred, c="xkcd:black", alpha=0.5)
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
    ax0.set_title("Total Interaction Energy")
    
    ax1 = plt.subplot2grid((2,4), (0,2))
    ax1.scatter(elec, elec_pred, c="xkcd:red", s=7)
    lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]),
            np.max([ax1.get_xlim(), ax1.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax1.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax1.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax1.text(0.1,0.9,"MAE = %.2f kcal/mol"%(elst_mae),
                fontsize=10,transform=ax1.transAxes)
    ax1.grid(color='k',linestyle=':',alpha=0.4)
    ax1.set_title("Electrostatics")

    ax2 = plt.subplot2grid((2,4), (0,3))
    ax2.scatter(exch, exch_pred, c="xkcd:green", s=7)
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax2.set_aspect('equal')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax2.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax2.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax2.text(0.1,0.9,"MAE = %.2f kcal/mol"%(exch_mae),
                fontsize=10,transform=ax2.transAxes)
    ax2.grid(color='k',linestyle=':',alpha=0.4)
    ax2.set_title("Exchange")

    ax3 = plt.subplot2grid((2,4), (1,2))
    ax3.scatter(ind, ind_pred, c="xkcd:blue", s=7)
    lims = [np.min([ax3.get_xlim(), ax3.get_ylim()]),
            np.max([ax3.get_xlim(), ax3.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax3.set_aspect('equal')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax3.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax3.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax3.text(0.1,0.9,"MAE = %.2f kcal/mol"%(ind_mae),
                fontsize=10,transform=ax3.transAxes)
    ax3.grid(color='k',linestyle=':',alpha=0.4)
    ax3.set_title("Induction")

    ax4 = plt.subplot2grid((2,4), (1,3))
    ax4.scatter(disp, disp_pred, c="xkcd:orange", s=7)
    lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]),
            np.max([ax4.get_xlim(), ax4.get_ylim()])]
    lims_p1 = [lims[0], lims[1]-1]
    lims_m1 = [lims[0]+1, lims[1]]
    ax4.set_aspect('equal')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.plot(lims,lims,'k-', alpha=0.5,zorder=0)
    ax4.plot(lims_p1,lims_m1,'k--', alpha=0.75,zorder=0)
    ax4.plot(lims_m1,lims_p1,'k--', alpha=0.75,zorder=0)
    ax4.text(0.1,0.9,"MAE = %.2f kcal/mol"%(disp_mae),
                fontsize=10,transform=ax4.transAxes)
    ax4.grid(color='k',linestyle=':',alpha=0.4)
    ax4.set_title("Dispersion")

    #plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
    fig.text(0.5, 0.1, "SAPT0 Interaction Energy (kcal/mol)", ha='center')
    fig.text(0.1, 0.5, "NN Predicted Interaction Energy (kcal/mol)", 
                va="center", rotation="vertical")
    plt.suptitle(plot_title, x=0.5, y=0.9, ha="center")
    plt.subplots_adjust(wspace=0.4, hspace=0.0)
    plt.show()
    return


plot_sapt_from_file("./test_results/neutral-SSI_0.0125-spike_100-100-75Acc--NMe-acetamide_Don--Uracil-PDB--xyzcoords-nrgs-dmv2-format.csv",
                    "Intermolecular BPNN Tested on Crystallographic NMe-acetamide / N-Isopropylethanimidic acid")
#plot_sapt_from_folder("./test_results/0.0125",
#                    "Intermolecular BPNN Tested on Crystallographic NMe-Acetamide / X Configurations")
#plot_sapt_from_file("./NMA-Aniline-0.1-crystal-pert.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Unlabeled Perturbed Dimers")
#plot_sapt_from_file("./artif_and_labeled_pert_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data and Labeled Perturbed Dimers")
#plot_sapt_from_file("./symfun_experiment_crystal.csv",
#                    "NMA-Aniline Crystallographic Test Trained with Artificial Data, Labeled Perturbed Dimers, and 'Intermolecular' Symmetry Functions")
