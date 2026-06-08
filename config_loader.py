import yaml
import itertools
import numpy as np
from particle_model import build_particle
from angular_formula import get_angle_formula
import math

def load_config(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename) as f:
        ret = yaml.safe_load(f)
    return ret


class Particle:
    def __init__(self, name, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __str__(self):
        return self.name


def s_range(l1, l2):
    a = l1
    while a < l2:
        yield a
        a += 1

class Decay:
    def __init__(self, core, outs, **kwargs):
        self.core = core
        self.outs = outs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_ls_list(self):
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        pa = getattr(self.core, "P", None)
        pb = getattr(self.outs[0], "P", None)
        pc = getattr(self.outs[1], "P", None)
        ls_list = []
        for s in s_range(abs(jb-jc), jb+jc +1):
            for l in range(int(abs(ja-s)), int(ja + s)+1):
                if not getattr(self, "p_break", False):
                    if (pa is not None and pb is not None and pc is not None):
                        if l % 2 == (1 if pa*pb*pc == 1 else 0):
                            continue
                ls_list.append((l,s))
        return ls_list

    def get_ls_names(self):
        prefix = str(self).replace("+", ".")
        ret = []
        for i in range(len(self.get_ls_list())):
            ret.append(f"{prefix}_g_ls_{i}")
        return ret

    def __str__(self):
        return f"{self.core}->" + "+".join([str(i) for i in self.outs])

class DecayChain:
    def __init__(self, decays):
        self.decays = decays
        decay_particles = [i.core.name for i in self.decays]
        out_particles = []
        for i in self.decays:
            for j in i.outs:
                out_particles.append(j.name)
        self.finals = [i for i in out_particles if i not in decay_particles]
        self.top = [i for i in decay_particles if i not in out_particles][0]
        self.inner = [i for i in out_particles if i in decay_particles]

    def get_ls_combination(self):
        ls_lists = [i.get_ls_list() for i in self.decays]
        # print(ls_lists)
        ret = list(itertools.product(*ls_lists))
        return ret

    def get_gls_combination(self):
        ls_lists = [i.get_ls_names() for i in self.decays]
        # print(ls_lists)
        total = str(self) + "_total_0"
        ret = list(itertools.product([total], *ls_lists))
        return ret

    def get_topo_map(self):
        topo_map = {k: [k] for k in self.finals}
        while self.top not in topo_map:
            for j in self.decays:
                if all([k.name in topo_map for k in j.outs]):
                    tmp = []
                    for k in j.outs:
                        tmp += topo_map[k.name]
                    topo_map[j.core.name] = tmp
        return topo_map

    def topo_id(self):
        topo_map = self.get_topo_map()
        topo_id = tuple(sorted([tuple(sorted(topo_map[i]))  for i in self.inner   ]))
        return topo_id

    def __str__(self):
        names = [str(i) for i in self.decays]
        return "".join(names)



class DecayGroup:
    def __init__(self, chains):
        self.chains = chains

    def __iter__(self):
        return iter(self.chains)

    def get_partial_waves(self):
        ret = []
        for i in self.chains:
            for j in i.get_ls_combination():
                ret.append((j, i))
        return ret

    def get_partial_waves_params(self):
        ret = []
        for i in self.chains:
            for j in i.get_gls_combination():
                ret.append(j)
        return ret



class Config:
    def __init__(self, filename):
        self.dic = load_config(filename)
        top = self.dic["particle"]["$top"]
        finals = list(self.dic["particle"]["$finals"])
        self.top = top
        self.finals = finals

        self.decay_struct = self.build_decay_struct(self.dic["decay"], top, finals)
        self.decay_chains_lst = self.get_decay_chains(self.decay_struct, self.dic["particle"])
        self.full_decay = self.build_decay_chains(self.decay_chains_lst, self.dic["particle"])

        self.n_decay = len(self.decay_struct[0])
        self.n_res = self.n_decay - 1
        self.n_angles = 2 * self.n_decay
        if self.dic["particle"][top]["J"] == 0:
            self.n_angles = self.n_angles - 3 # 3d rotaion is not need
        self.topo_index = self.get_topo_index(self.full_decay)
        self.n_topo = len(self.topo_index)
        self.m0_phys_name = []
        self.g0_phys_name = []
        self.unique_l = []
        self.unique_bw = []
        self.unique_gamma = []
        self.unique_fl = []
        self.unique_angle_basis = []

    def get_topo_index(self, decay):
        topo_id = {}
        for i in decay.chains:
            tmp = i.topo_id()
            if tmp not in topo_id:
                topo_id[tmp] = len(topo_id)
        return topo_id


    def build_decay_struct(self, decay, top, finals):
        # loop to find the chains that top -> a + b, a-> ..., b -> ...
        # filter with finals

        def get_sub_decays(dic, top):

            if top not in dic:
                return [[]]
            res_decay = dic[top]
            if not isinstance(res_decay[0], list):
                res_decay = [ res_decay ]

            ret = []
            for i in res_decay:
                outs = [j for j in i if isinstance(j, str)]
                kwargs_list = [k for k in i if isinstance(k, dict)]
                kwargs = {}
                for kw in kwargs_list:
                    kwargs.update(kw)
                outdecay = [get_sub_decays(dic, outi) for outi in outs]
                # print(outdecay)
                for prod in itertools.product(*outdecay):
                    # print("prod", "prod")
                    table = [(top, outs, kwargs)]
                    for i in prod:
                        table += i
                    ret.append(table)

            return ret

        return get_sub_decays(decay, top)


    def get_decay_chains(self, decay_struct, res_map):
        ret = []
        for decay_chain in decay_struct:
            used_res = []
            for i in decay_chain:
                if i[0] not in used_res:
                        used_res.append(i[0])
                for j in i[1]:
                    if j not in used_res:
                        used_res.append(j)
            combinations = []
            replace_res = []
            for j in used_res:
                if isinstance(res_map[j], list):
                    combinations.append(res_map[j])
                    replace_res.append(j)

            for c in itertools.product(*combinations):
                tmp = decay_chain
                for ci,ji in zip(c, replace_res):
                    tmp = replace_chain(tmp, ji, ci)
                ret.append(tmp)
        return ret

    def build_decay_chains(self, lst, dic):
        all_particles = []
        for i in lst:
            for j in i:
                if j[0] not in all_particles:
                    all_particles.append(j[0])
                for k in j[1]:
                    if k not in all_particles:
                        all_particles.append(k)
        particles = {}
        for k in all_particles:
            particles[k] = Particle(k, **dic[k])
            particles[k]._model = build_particle(k, **dic[k])
        ret = []
        for i in  lst:
            tmp = []
            for j in i:
                tmp.append(Decay(particles[j[0]], [particles[k] for k in j[1]], **j[2]))
            chain = DecayChain(tmp)
            ret.append(chain)
        return DecayGroup(ret)

    def get_max_mass_range(self):
        m_center = self.dic["particle"][self.top]["mass"]
        m_finals = sorted([self.dic["particle"][i]["mass"] for i in self.finals])
        m_max = m_center - m_finals[0]
        m_min = m_finals[0]+ m_finals[1]
        return m_min, m_max

    def build_gamma_table(self, n_interp=500):
        gamma_table = {}
        m_min, m_max = self.get_max_mass_range()
        m = np.linspace(m_min-0.01, m_max + 0.01, n_interp)
        for chain in self.full_decay.chains:
            for decay in chain.decays[1:]:
                g0 = decay.core._model.get_gamma_name()
                if g0[0] not in gamma_table:
                    gamma = decay.core._model.gamma(m)
                    for i,j in zip(g0, gamma):
                        gamma_table[i] = j
        return gamma_table, m[0], m[1]-m[0]


    def get_max_q_range(self):
        m_center = self.dic["particle"][self.top]["mass"]
        m_finals = sorted([self.dic["particle"][i]["mass"] for i in self.finals])
        m_max = m_center - m_finals[0]
        m1, m2 = m_finals[0:2]
        q_max = math.sqrt( (m_max**2 - (m1+m2)**2)*(m_max**2-(m1+m2)**2) )/2/m_max
        return 0., q_max

    def build_fl_table(self, l_list, n_interp=500):
        ret = []
        d = 3.0
        q_min, q_max = self.get_max_q_range()
        q = np.linspace(q_min, q_max, n_interp)
        for l in l_list:
            if l == 0:
                ret.append(np.ones_like(q))
            elif l == 1:
                z = (q * d)**2
                ret.append( q / np.sqrt( 1 + z  ))
            elif l == 2:
                z = (q * d)**2
                ret.append( q**2 / np.sqrt( 9 + 3*z + z**2 ))
            else: # not implemeted
                ret.append(np.ones_like(q))
        return np.stack(ret, axis=0), q[0], q[1]-q[0]





    def build_single_index(self):
        topo_id_map = self.topo_index
        print(topo_id_map)

        bw_gamma = {}

        for ls, decaychain in self.full_decay.get_partial_waves():
            for li in ls:
                if li[0] not in self.unique_l:
                    self.unique_l.append(li[0])
            topo = topo_id_map[decaychain.topo_id()]
            for idx, decay in enumerate(decaychain.decays):
                if idx != 0: # not top decay
                    m_name = decay.core.name +"_mass"
                    if m_name not in self.m0_phys_name:
                        self.m0_phys_name.append(m_name)
                    m_idx = self.n_res*topo+idx-1
                    # print(decaychain, topo, idx,m_idx)
                    bw_id = (m_name, m_idx)
                    if bw_id not in self.unique_bw:
                        self.unique_bw.append(bw_id)
                    tmp = []
                    for g0 in decay.core._model.get_gamma_name():
                        if g0 not in self.g0_phys_name:
                            self.g0_phys_name.append(g0)
                        g_id = (g0, m_idx)
                        if g_id not in self.unique_gamma:
                            self.unique_gamma.append(g_id)
                        tmp.append(g_id)
                    bw_gamma[bw_id] = tmp
                fl_id = (ls[idx][0], self.n_decay*topo+idx)
                if fl_id not in self.unique_fl:
                    self.unique_fl.append(fl_id)
            ang_formula = get_angle_formula(decaychain, ls)
            for ang in ang_formula:
                basis_key = (topo, tuple(ang["k"]),tuple(ang["b"]))
                if basis_key not in self.unique_angle_basis:
                    self.unique_angle_basis.append(basis_key)
        # build ret
        ret = {}
        matrix_gamma = np.zeros((len(self.unique_gamma), len(self.unique_bw)))
        for idx, k in enumerate(self.unique_bw):
            for j in bw_gamma[k]:
                matrix_gamma[self.unique_gamma.index(j), idx] = 1.0
        ret["matrix_gamma"] = matrix_gamma
        bw_order = []
        fl_order = []
        matrix_angle = []
        for ls, decaychain in self.full_decay.get_partial_waves():
            topo = topo_id_map[decaychain.topo_id()]
            for idx, decay in enumerate(decaychain.decays):
                if idx != 0: # not top decay
                    m_name = decay.core.name +"_mass"
                    m_idx = self.n_res*topo+idx-1
                    bw_id = (m_name, m_idx)
                    bw_order.append(self.unique_bw.index(bw_id))
                fl_id = (ls[idx][0], self.n_decay*topo+idx)
                fl_order.append(self.unique_fl.index(fl_id))
            ang_formula = get_angle_formula(decaychain, ls)
            matrix_angle_tmp = np.zeros(len(self.unique_angle_basis))+0j
            for ang in ang_formula:
                coeff = ang["coeffs"]
                basis_key = (topo, tuple(ang["k"]), tuple(ang["b"]))
                matrix_angle_tmp[self.unique_angle_basis.index(basis_key)] = coeff
            matrix_angle.append(matrix_angle_tmp)
        ret["matrix_angle"] = np.stack(matrix_angle, axis=-1)
        ret["bw_order"] = np.array(bw_order)
        ret["fl_order"] = np.array(fl_order)

        angle_index = []
        angle_k = []
        angle_b = []
        for key in self.unique_angle_basis:
            angle_index.append(key[0])
            angle_k.append(key[1])
            angle_b.append([(0 if k == "cos" else -np.pi/2) for k in key[2]])
        ret["angle_index"] = np.stack(angle_index)
        ret["angle_k"] = np.stack(angle_k)
        ret["angle_b"] = np.stack(angle_b)

        m0_index = []
        mass_index = []
        for key in self.unique_bw:
            m0_index.append(self.m0_phys_name.index(key[0]))
            mass_index.append(key[1])
        ret["mass_index"] = np.stack(mass_index)
        ret["m0_index"] = np.stack(m0_index)
        g0_index = []
        g0_mass_index = []
        for key in self.unique_gamma:
            g0_index.append(self.g0_phys_name.index(key[0]))
            g0_mass_index.append(key[1])
        ret["g0_mass_index"] = np.stack(g0_mass_index)
        ret["g0_index"] = np.stack(g0_index)
        fl_type = []
        fl_q_index = []
        for key in self.unique_fl:
            fl_type.append(self.unique_l.index(key[0]))
            fl_q_index.append(key[1])
        ret["fl_type"] = np.stack(fl_type)
        ret["fl_q_index"] = np.stack(fl_q_index)

        gamma_table, g_min, g_delta = self.build_gamma_table()
        ret["gamma_table"] = np.stack([gamma_table[i] for i in self.g0_phys_name], axis=0)
        ret["gamma_min"] = g_min
        ret["gamma_delta"] = g_delta
        ret["fl_table"], ret["fl_min"], ret["fl_delta"] = self.build_fl_table(self.unique_l)

        return ret

    def build_all_index(self):
        # loop and shift based on block
        base = self.build_single_index()
        ck = self.full_decay.get_partial_waves_params()


        def _repeat(arr, count=8):
            return np.concatenate([arr]*count, axis=0)

        def _shift_repeat(arr, strip, count=8):
            all_arr = []
            for i in range(count):
                all_arr.append(arr + strip * i)
            return np.concatenate(all_arr,axis=0)

        def _matrix_repeat(arr, count=8):
            all_arr = []
            for i in range(count):
                all_tmp_arr = []
                for j in range(count):
                    if i==j:
                        all_tmp_arr.append(arr)
                    else:
                        all_tmp_arr.append(np.zeros_like(arr))
                all_arr.append(np.concatenate(all_tmp_arr, axis=1))
            return np.concatenate(all_arr,axis=0)

        ret = {}
        ret["m0_index"] = _repeat(base["m0_index"])
        ret["g0_index"] = _repeat(base["g0_index"])
        ret["fl_type"] = _repeat(base["fl_type"])
        ret["angle_k"] = _repeat(base["angle_k"])
        ret["angle_b"] = _repeat(base["angle_b"])
        ret["fl_type"] = _repeat(base["fl_type"])
        ret["mass_index"] = _shift_repeat(base["mass_index"], self.n_topo* self.n_res)
        ret["g0_mass_index"] = _shift_repeat(base["g0_mass_index"], self.n_topo* self.n_res)
        ret["fl_q_index"] = _shift_repeat(base["fl_q_index"], self.n_topo* self.n_decay)
        ret["bw_order"] = _shift_repeat(base["bw_order"], len(self.unique_bw))
        ret["fl_order"] = _shift_repeat(base["fl_order"], len(self.unique_fl))
        ret["angle_index"] = _shift_repeat(base["angle_index"], self.n_topo)
        ret["matrix_angle"] = _matrix_repeat(base["matrix_angle"])
        ret["matrix_gamma"] = _matrix_repeat(base["matrix_gamma"])

        for name in ["gamma_table","fl_table", "gamma_min", "gamma_delta", "fl_min", "fl_delta"]:
            ret[name] = base[name]

        return ret

    def get_ck_map(self):
        cks = self.full_decay.get_partial_waves_params()
        ret = []
        for i in range(8):
            for j in cks:
                if i < 4:
                    ret.append(j)
                else:
                    ret.append((j[0], j[1].replace("g_ls", "g_lsbar"), *j[2:]))
        return ret




def replace_chain(chain, res, res2):
    ret = []
    for decay in chain:
        core = res2 if decay[0] == res else decay[0]
        outs = [res2 if i == res else i for i in decay[1]]
        ret.append( (core, outs, *decay[2:]))
    return ret


if __name__=="__main__":
    a = Config("config_angle.yml")
    c = a.build_all_index()
    from numpy_kernel import NumpyKernel
    kernel  = NumpyKernel(c)
    ck = a.get_ck_map()
    # print(c)
    n_events = 7
    n = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }
    )
    l = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    },
    norm=1.0
    )

    params = {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    data =     {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }


    l = kernel._compute(params,
    data,
    norm=1.0
    )
    l2 = [kernel._compute(params,
    {k: v[i:i+1] for k, v in data.items()},
    norm=1.0
    ) for i in range(n_events)]
    print(l[0], sum(i[0] for i in l2))
    print(l[2], [i[2] for i in l2])
