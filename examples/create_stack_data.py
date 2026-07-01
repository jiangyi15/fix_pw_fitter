from tf_pwa.config_loader import ConfigLoader
import sys
sys.path.insert(0, "../../ana/test_4pi/test_full_gpu")
import extra_amp
import tensorflow as tf
from tf_pwa.data import data_index
import numpy as np
config = ConfigLoader("config_angle.yml")

order = {(('pim1', 'pip1'), ('pim2', 'pip2')): 0, (('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')): 1, (('pim1', 'pim2', 'pip1'), ('pim1', 'pip1')): 2}


def topo_id(chain):
    lst = chain.topology_id()
    lst = [i for i in lst if len(i) not in [1,4]]
    # print(lst)
    lst = tuple(sorted( tuple(sorted(i)) for i in lst))
    return lst


def read_single_data(data):

    decay_chains = data["decay"].keys()
    mass = [None] * len(decay_chains)
    q = [None] * len(decay_chains)
    angle = [None] * len(decay_chains)

    for decay_chain in decay_chains:
        idx = order[topo_id(decay_chain)]
        tmp_mass = []
        tmp_q = []
        for decay in decay_chain:
            if decay.core != decay_chain.top:
                tmp_mass.append( data["particle"][decay.core]["m"].numpy() )
            tmp_q.append( np.sqrt(data["decay"][decay_chain][decay]["|q|2"].numpy() ))
        mass[idx] = np.stack(tmp_mass, axis=-1)
        q[idx] = np.stack(tmp_q, axis=-1)
        ang1 = data["decay"][decay_chain][decay_chain[1]][decay_chain[1].outs[0]]["ang"]
        ang2 = data["decay"][decay_chain][decay_chain[2]][decay_chain[2].outs[0]]["ang"]

        if idx == 0:
            phi = ang1["alpha"].numpy() + ang2["alpha"].numpy()
        else:
            phi = ang2["alpha"].numpy()
        theta1 = ang1["beta"].numpy()
        theta2 = ang2["beta"].numpy()
        angle[idx] = np.stack([phi, theta1, theta2], axis=-1)
    return mass, q, angle

def read_id_data(data):
    mass, q, angle = read_single_data(data)
    for k, v in data["id_swap"].items():
        print(k)
        tmp_mass, tmp_q, tmp_angle = read_single_data(v)
        mass += tmp_mass
        q += tmp_q
        angle += tmp_angle
    return mass, q, angle

def read_cp_data(data):
    mass, q, angle = read_id_data(data)
    tmp_mass, tmp_q, tmp_angle = read_id_data(data["cp_swap"])
    mass += tmp_mass
    q += tmp_q
    angle += tmp_angle
    return np.stack(mass, axis=-2), np.stack(q, axis=-2), np.stack(angle, axis=-2)


def read_data(data):
    print(data.keys())
    mass, q, angle = read_cp_data(data)
    frac = np.where(data["tag1"]==0,0.5, np.where(data["tag1"]>0, 1-data["eta1"], data["eta1"]))
    time = data["time"]
    bkg = data["bg_value"]
    weight = data["weight"]
    return {"mass": mass, "q": q, "angles": angle, "frac": frac, "time": time, "bkg_raw": bkg, "weight": weight}

with tf.device("CPU"):
    # data  = config.get_data("data")[0]
    # var= read_data(data)
    # np.savez("data_arrays.npz", **var)
    # data  = config.get_data("phsp")[0]
    # var= read_data(data)
    # np.savez("phsp_arrays.npz", **var)
    data  = config.get_data("phsp_noeff_sym")[0]
    var= read_data(data)
    np.savez("data/phsp_noeff_sym_arrays.npz", **var)
