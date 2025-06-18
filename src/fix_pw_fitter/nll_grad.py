import numpy as np

class AbsParamsConverter:
    def random_x(self):
        pass
    def build_ck(self, x):
        pass
    def build_ck_vjp(self, x, ck, g):
        pass

class ParamsConverter(AbsParamsConverter):
    def __init__(self, combinations, fixed_params={}):
        self.combinations = combinations
        self.fixed_params = fixed_params
        self.all_used_x = []
        for i in self.combinations:
            for j in i:
                if j not in self.all_used_x and j not in self.fixed_params:
                    self.all_used_x.append(j)

    def random_x(self):
        return np.random.random((len(self.all_used_x),2)).reshape((-1,))

    def build_ck(self, x):
        ret = []
        x = x[::2] + 1.0j*x[1::2]
        all_x = {name: x[i] for i, name in enumerate(self.all_used_x)}
        for i in self.combinations:
            tmp = 1.0+0j
            for j in i:
                if j in self.fixed_params:
                    tmp = tmp*self.fixed_params[j]
                else:
                    tmp = tmp*all_x[j]
            ret.append(tmp)
        return np.stack(ret)

    def build_ck_vjp(self, x, ck, g):
        x = x[::2] + 1.0j*x[1::2]
        all_x = {name: x[i] for i, name in enumerate(self.all_used_x)}
        ret = {name: 0.0j for name in self.all_used_x}
        for i, comb in enumerate(self.combinations):
            gi = g[i]
            scale = 1.0+0j
            for j in comb:
                if j in self.fixed_params:
                    scale *= self.fixed_params[j]

            for j, ji in enumerate(comb):
                if ji in self.all_used_x:
                    other_prod = 1.+0j
                    for k, ki in enumerate(comb):
                        if k!=j and ki in self.all_used_x:  # allow same ji,ki, but different k,j
                            other_prod = other_prod * all_x[k]
                    ret[ji] += scale * other_prod* gi
        complex_j = np.stack([ret[name] for name in self.all_used_x])
        return np.stack( [2* np.real(complex_j),
                          - 2* np.imag(complex_j)], axis=-1).reshape((-1,))


class NLLGrad:
    def __init__(self, Fijk, wi, M, pc: AbsParamsConverter):
        self.Fijk = Fijk
        self.wi = wi
        self.Mkl = M
        self.nsig = np.sum(wi)
        self.pc = pc

    def nll_grad_lnL_ck(self, ck):
        Aij = np.sum(ck*self.Fijk, axis=-1)
        Aij_C = np.conj(Aij)
        Pi = np.sum(np.abs(Aij)**2, axis=-1) # np.sum(Aij*Aij_C, axis=-1)
        lnLi = np.log(np.real(Pi))
        lnL = np.sum(self.wi*lnLi)
        dlnLidck = np.sum(self.Fijk *Aij_C[...,np.newaxis], axis=-2)/Pi[:,None]
        dlnLdck = np.sum(self.wi[...,np.newaxis]*dlnLidck, axis=0)
        return lnL, dlnLdck

    def nll_grad_lnN_ck(self, ck):
        dNdck = np.sum(self.Mkl * np.conj(ck), axis=-1)
        N = np.sum(ck * dNdck)
        dlnNdck = dNdck/N
        return np.log(np.real(N)), dlnNdck

    def nll_grad_ck(self, ck):
        lnL, dlnLdck = self.nll_grad_lnL_ck(ck)
        lnN, dlnNdck = self.nll_grad_lnN_ck(ck)
        ret = -lnL + self.nsig * lnN
        ret_g = - dlnLdck + self.nsig * dlnNdck
        return ret, ret_g

    def nll_grad(self, x):
        ck = self.pc.build_ck(x)
        lnL, dlnLdck = self.nll_grad_ck(ck)
        dlnLdx = self.pc.build_ck_vjp(x, ck, dlnLdck)
        return lnL, dlnLdx

    def minimize(self, x0=None):
        if x0 is None:
            x0 = self.pc.random_x()
        from scipy.optimize import minimize
        ret = minimize(self.nll_grad, x0, jac=True)
        return ret
