import numpy as np

ALL_MODELS = {
}

def register_model(name):
    def _f(cls):
        global ALL_MODELS
        ALL_MODELS[name] = cls
        return cls
    return  _f

def build_particle(name, **kwargs):
    model = kwargs.pop("model", "BW")
    return ALL_MODELS[model](name, **kwargs)

@register_model("BW")
class BaseModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def get_gamma_count(self):
        return 1

    def get_gamma_name(self):
        return [f"{self.name}_width"]

    def gamma(self, m):
        return [np.ones_like(m) + 0j]


@register_model("one")
class OneModel(BaseModel):
    def gamma(self, m):
        # 1 = 1/(m0**2 - m**2 - im0 g0 Gamma)
        # gamma = i (1/1 - m0**2 + m**2)/m0/g0
        m0 = self.kwargs["mass"]
        g0 = self.kwargs.get("width", 1.0)
        return [1j * (1 - m0**2 + m**2 )/m0/g0]


@register_model("FlatteC")
class OneModel(BaseModel):

    def get_gamma_count(self):
        return len(self.kwargs["mass_list"])

    def get_gamma_name(self):
        return [f"{self.name}_g{i}" for i in range(self.get_gamma_count())]

    def gamma(self, m):
        return [np.ones_like(m)  + 0j] * self.get_gamma_count()

@register_model("GS_rho")
class OneModel(BaseModel):
    def gamma(self, m):
        return [np.ones_like(m)  + 0j] * self.get_gamma_count()

@register_model("Bugg")
class OneModel(BaseModel):
    def gamma(self, m):
        return [np.ones_like(m)  + 0j] * self.get_gamma_count()

@register_model("width_linear_npy")
class OneModel(BaseModel):

    def gamma(self, m):
        data = np.load(self.kwargs["file"])
        mi = data[:,0]
        fi = data[:,1] + 1j * data[:,2]
        y = np.interp(m, mi,fi)
        y0 = np.interp(self.kwargs["mass"], mi,fi)
        return [ y/y0 ]
