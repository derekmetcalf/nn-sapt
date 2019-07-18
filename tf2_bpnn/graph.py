import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Run as: \n >>> python graph.py [filename]')
        print('Where file \'filename\' contains plaintext output of training run')
        exit()
    filename = sys.argv[1]

    with open(filename, 'r') as fo:
        lines = fo.readlines()
        lines = [line for line in lines if line.startswith('Epoch')]
        lines = [line.split() for line in lines]

    RMSE_t = np.array([float(line[4]) for line in lines])
    RMSE_v = np.array([float(line[6]) for line in lines])
    MAE__t = np.array([float(line[9]) for line in lines])
    MAE__v = np.array([float(line[11]) for line in lines])

    k = 50  # smoothing param
    N = len(RMSE_t)

    RMSE_t2 = np.zeros(N - k)
    RMSE_v2 = np.zeros(N - k)
    MAE__t2 = np.zeros(N - k)
    MAE__v2 = np.zeros(N - k)

    for i in range(N - k):
        RMSE_t2[i] = np.median(RMSE_t[i:(i + k)])
        RMSE_v2[i] = np.median(RMSE_v[i:(i + k)])
        MAE__t2[i] = np.median(MAE__t[i:i + k])
        MAE__v2[i] = np.median(MAE__v[i:i + k])

    fig = plt.figure()

    epochs = np.arange(1, N + 1)

    plt.subplot(2, 2, 1)
    plt.plot(epochs, RMSE_t, label='Train')
    plt.plot(epochs, RMSE_v, label='Val')
    plt.title('RMSE')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, MAE__t, label='Train')
    plt.plot(epochs, MAE__v, label='Val')
    plt.title('MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 3)
    plt.plot(epochs[k:], RMSE_t2, label='Train')
    plt.plot(epochs[k:], RMSE_v2, label='Val')
    plt.title(f'RMSE Rolling Median ({k} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 4)
    plt.plot(epochs[k:], MAE__t2, label='Train')
    plt.plot(epochs[k:], MAE__v2, label='Val')
    plt.title(f'MAE Rolling Median ({k} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.show()

    plt.subplot(2, 2, 1)
    plt.axhline(0, color='black')
    plt.plot(epochs[1:], np.diff(RMSE_t), label='Train')
    plt.plot(epochs[1:], np.diff(RMSE_v), label='Val')
    plt.title('$\\Delta$ RMSE')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 2)
    plt.axhline(0, color='black')
    plt.plot(epochs[1:], np.diff(MAE__t), label='Train')
    plt.plot(epochs[1:], np.diff(MAE__v), label='Val')
    plt.title('$\\Delta$ MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 3)
    plt.axhline(0, color='black')
    plt.plot(epochs[k+1:], np.diff(RMSE_t2), label='Train')
    plt.plot(epochs[k+1:], np.diff(RMSE_v2), label='Val')
    plt.title(f'$\\Delta$ RMSE Rolling Median ({k} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.subplot(2, 2, 4)
    plt.axhline(0, color='black')
    plt.plot(epochs[k+1:], np.diff(MAE__t2), label='Train')
    plt.plot(epochs[k+1:], np.diff(MAE__v2), label='Val')
    plt.title(f'$\\Delta$ MAE Rolling Median ({k} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (kcal / mol)')
    plt.xlim(0,N)

    plt.show()



    #plt.subplot(2, 2, 3)
    #plt.plot(np.diff(RMSE_t), label='Train')
    #plt.plot(np.diff(RMSE_v), label='Val')
    #plt.title('\Delta RMSE')
    #plt.legend()
    #plt.xlabel('Epochs')
    #plt.ylabel('Error (kcal / mol)')

    #plt.subplot(2, 2, 4)
    #plt.plot(np.diff(MAE__t), label='Train')
    #plt.plot(np.diff(MAE__v), label='Val')
    #plt.title('\Delta MAE')
    #plt.xlabel('Epochs')
    #plt.ylabel('Error (kcal / mol)')

