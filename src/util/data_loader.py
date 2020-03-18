import numpy as np
import os
from datetime import datetime

from src.ising_gt import ising_ground_truth
from src.util.helper import make_locally_connect

def load_data(cf):
    size = np.prod(cf.input_size)
    if cf.pb_type == "maxcut":
        laplacian_data_path = "./data/maxcut/graph{}.npy".format(cf.input_size)
        if not os.path.exists(laplacian_data_path):
            laplacian = np.random.randint(2, size=[size,size])
            laplacian = (laplacian + laplacian.transpose())//2
            np.fill_diagonal(laplacian, 0)
            np.save(laplacian_data_path, laplacian)

            f=open("results.txt", "a+")
            f.write("[Date:{}] Maxcut {}\n".format(datetime.now().strftime("%m%d_%H%M%S"), cf.input_size))

            if size < 23:
                quant, state, time_ellapsed = ising_ground_truth(cf, laplacian, fig_save_path=laplacian_data_path[:-4]+".png")
                f.write("Ground Truth - Edges cut: {}, Time: {} seconds\n".format(quant, time_ellapsed))
                f.write("Optimal State: {}\n".format(state))
                print('Ground Truth')
                print("Cut size: {}, Time ellapsed {}".format(quant, time_ellapsed))

            from src.offshelf.maxcut import off_the_shelf
            quant, time_ellapsed = off_the_shelf(cf, laplacian, method="random_cut")
            f.write("Random Cut - Edges cut: {}, Time: {} seconds\n".format(quant, time_ellapsed))

            quant, time_ellapsed = off_the_shelf(cf, laplacian, method="greedy_cut")
            f.write("Greedy Cut - Edges cut: {}, Time: {} seconds\n".format(quant, time_ellapsed))

            quant, time_ellapsed = off_the_shelf(cf, laplacian, method="goemans_williamson")
            f.write("Goemans Williamson - Edges cut: {}, Time: {} seconds\n".format(quant, time_ellapsed))
            f.write("----------------------------------------------------------------------------------------\n")
            f.close()
        else:
            laplacian = np.load(laplacian_data_path)
        return laplacian
    elif cf.pb_type == "spinglass":
        J_data_path = "./data/spinglass/J{}.npy".format(cf.input_size)
        if not os.path.exists(J_data_path):
            J_mtx = np.random.normal(0,0.5,size**2)
            J_mtx = np.reshape(J_mtx, [size,size])
            J_mtx = (J_mtx + J_mtx.transpose())/2
            J_mtx = make_locally_connect(cf, J_mtx)
            np.fill_diagonal(J_mtx, 0)
            np.save(J_data_path, J_mtx)

            if J_mtx.shape[0] < 30:
                quant, state, time_ellapsed = ising_ground_truth(cf, J_mtx, fig_save_path=J_data_path[:-4]+".png")
                f=open("results.txt", "a+")
                f.write("[Date:{} - Ground Truth] Spinglass {}\n".format(datetime.now().strftime("%m%d_%H%M%S"), cf.input_size))
                f.write("Time: {} seconds, Edges cut: {}\n".format(time_ellapsed, quant))
                f.write("Optimal State: {}\n".format(state))
                f.write("----------------------------------------------------------------------------------------\n")
                f.close()
        else:
            J_mtx = np.load(J_data_path)
        return J_mtx