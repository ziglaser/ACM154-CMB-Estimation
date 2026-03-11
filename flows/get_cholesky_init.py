import numpy as np

def get_init_cosmology():
    #this covariance is copied and pasted from a terminal printout of a NUTS warmup.
    cov = np.array([[ 8.94381638e+01,  1.38002476e-02, -9.95204821e-02],
    [ 1.38002476e-02,  1.38736805e-05, -6.41395900e-06],
    [-9.95204821e-02, -6.41395673e-06,  1.84258432e-04]])

    L = np.linalg.cholesky(cov)

    for i in range(3):
        cholesky_diag = np.log(L[i,i])
        print(f"diagonal {i}", cholesky_diag)

    print()
    print("cholesky L", L)

    idxs = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)] #the sequence of cholesky entries as used in the VI code

    lparams = []
    for id in idxs:
        if id[0] == id[1]:
            lparams.append(np.log(L[id[0],id[1]]))
        else:
            lparams.append(L[id[0],id[1]])
    print("lparams", lparams)
    np.save("init_lparams.npy", np.array(lparams))

def get_init_toy():
    #this covariance is copied and pasted from a terminal printout of a NUTS warmup.
    cov = np.array([[0.48898393, 0.06150292], [0.06150291, 0.6299142 ]])
    L = np.linalg.cholesky(cov)

    for i in range(2):
        cholesky_diag = np.log(L[i,i])
        print(f"diagonal {i}", cholesky_diag)

    print()
    print("cholesky L", L)

    idxs = [(0,0), (1,0), (1,1)] #the sequence of cholesky entries as used in the VI code

    lparams = []
    for id in idxs:
        if id[0] == id[1]:
            lparams.append(np.log(L[id[0],id[1]]))
        else:
            lparams.append(L[id[0],id[1]])
    print("lparams", lparams)
    np.save("toy_init_lparams.npy", np.array(lparams))

get_init_toy()