from fix_pw_fitter.nll_grad import NLLGrad, ParamsConverter
import numpy as np

def test_nll_grad():
    combination = [
        ["a1"],
        ["a2"],
        ["a3"]
    ]

    pc = ParamsConverter(combination, {"a1": 1.0+0j})

    f1 = np.random.random((100,1,3)) + 1j*np.random.random((100,1,3))
    w = np.ones((100,))
    f2 = np.random.random((1000,1,3)) + 1j*np.random.random((1000,1,3))

    M = np.sum(np.sum( f2[:,:,:,None] * np.conj(f2[:,:,None,:]), axis=0 ),axis=0)

    nll = NLLGrad(f1,w, M, pc)

    x = pc.random_x()
    y, yg = nll.nll_grad(x)
    eps = 1e-5
    gall = []
    for i in range(x.shape[0]):
        x[i] += eps
        yi1, _ = nll.nll_grad(x)
        x[i] -= 2*eps
        yi2, _ = nll.nll_grad(x)
        x[i] += eps
        gi = (yi1-yi2)/(2*eps)
        gall.append(gi)
    gall = np.stack(gall)
    assert np.allclose(yg, gall)
