import numpy as np

class NumpyKernel:
    def __init__(self, config):
        self.m0_index = config["m0_index"]
        self.g0_index = config["g0_index"]
        self.fl_type = config["fl_type"]
        self.angle_k = config["angle_k"]
        self.angle_b = config["angle_b"]
        self.fl_type = config["fl_type"]
        self.mass_index = config["mass_index"]
        self.g0_mass_index = config["g0_mass_index"]
        self.fl_q_index = config["fl_q_index"]
        self.bw_order = config["bw_order"]
        self.fl_order = config["fl_order"]
        self.angle_index = config["angle_index"]
        self.matrix_angle = config["matrix_angle"]
        self.matrix_gamma = config["matrix_gamma"]
        self.gamma_table = config["gamma_table"]
        self.fl_table = config["fl_table"]
        self.gamma_min = config["gamma_min"]
        self.fl_min = config["fl_min"]
        self.gamma_delta = config["gamma_delta"]
        self.fl_delta = config["fl_delta"]


        self.n_basis = self.angle_k.shape[0]
        self.n_angle = self.angle_k.shape[1]
        self.n_wave = self.matrix_angle.shape[1]
        self.n_res = self.bw_order.size//self.n_wave
        self.n_decay = self.fl_order.size//self.n_wave





    def _compute(self, params, data, norm=None):

        ck = params["ck"]
        m0 = params["m0"]
        g0 = params["g0"]
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]

        mass = data["mass"]
        momentum = data["q"]
        angle = data["angle"]
        frac = data["frac"]

        # bw
        g0_all = np.take(g0, self.g0_index)
        g0_m = np.take(mass, self.g0_mass_index, axis=-1)

        g = g0_all * self.interp(self.gamma_table, self.g0_index, g0_m, self.gamma_min, self.gamma_delta)
        g_bw = np.dot(g, self.matrix_gamma)
        m0_all = np.take(m0, self.m0_index)
        m0_m = np.take(mass, self.mass_index, axis=-1)
        bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        bw_dom_all = np.take(bw_dom, self.bw_order, axis=-1)
        bw_p = np.prod(np.reshape(bw_dom_all, (-1, self.n_wave, self.n_res)), axis=-1)

        # fl
        fl_q = np.take(momentum, self.fl_q_index, axis=-1)
        fl = self.interp(self.fl_table, self.fl_type, fl_q, self.fl_min, self.fl_delta)
        fl_all = np.take(fl, self.fl_order)
        fl_p = np.prod(np.reshape(fl_all, (-1, self.n_wave, self.n_decay)), axis=-1)

        # angle
        ang = np.take(angle, self.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * self.angle_k + self.angle_b), axis=-1)
        fa = np.dot(ka, self.matrix_angle)
        a = ck * 1/bw_p * fa * fl_p
        a = np.reshape(a, (-1, 2, self.n_wave//2))
        ap = np.sum(a[:,0], axis=-1)
        am = np.sum(a[:,1], axis=-1)
        eL= np.exp( -1j * data["time"] * (-Delta_m/2  - 1j * (Gamma +  Delta_Gamma/2)/2 ))
        eH= np.exp( -1j * data["time"] * (+Delta_m/2  - 1j * (Gamma -  Delta_Gamma/2)/2 ))
        gp = (eL + eH)/2
        gm = (eL - eH)/2

        poq = poq_rho * np.exp(1j * pop_phi)

        pb = np.abs( gp * ap + gm * poq * am  )**2
        pbbar = np.abs( gm/poq * ap + gp * am  )**2

        P = frac * pb * (1-A_p) + (1-frac) * pbbar * (1+A_p)
        if norm is None:
            Q = np.sum(data["weight"] * P)
        else:
            Q = -np.sum(data["weight"] *  np.log(P /norm + data["bkg"]))

        grads = {
            "ck": ...,
            "m0": ...,
            "g0": ...,
            "scalar": ...,
            "norm": ...,
        }
        return Q, grads, P


    def interp(self, table, types, x, xmin, xdelta):
        diff = (x - xmin)/xdelta
        xbin = np.floor(diff).astype(np.intp)
        n_bins = table.shape[-1]
        xbin = np.clip(xbin, 0, n_bins - 2)
        delta = diff - xbin
        idx = types * n_bins + xbin
        left = np.take(table.flatten(), idx)
        right = np.take(table.flatten(), idx + 1)
        return (right - left) * delta + left











