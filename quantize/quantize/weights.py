import numpy as np

weights = [
0.0018711995799,
0.0006513987900,
0.0008464435231,
0.0006303764530,
0.0006622965447,
0.0004021397035,
0.0004410696274,
0.0004114628536,
0.0003627797123,
0.0003454878169,
0.0002228170051,
0.0004379294114,
0.0002571978257,
0.0002501719573,
0.0002253487618,
0.0003448522766,
0.0003770928306,
0.0003578538307,
0.0001429267140,
0.0004940042272,
0.0004171476466,
0.0005779922357,
0.0002396436902,
0.0003321021795,
0.0003998000756,
0.0003601130447,
0.0002006033464,
0.0001655414380,
0.0003261315869,
0.0002264439535,
0.0001793571282,
0.0002632572141,
0.0002318821789,
0.0001526736014,
0.0002143646415,
0.0001725665788,
0.0002819601213,
0.0002218158770,
0.0004456869501,
0.0004002414352,
0.0007060594507,
0.0003115265571,
0.0003896020643,
0.0006217754562,
0.0003129696997,
0.0001657398242,
0.0003044340992,
0.0001685501629,
0.0002079607074,
0.0002146319020,
0.0001416871818,
0.0001819856406,
0.0001830206019,
0.0001369532692,
0.0001461390784,
0.0001794958079,
0.0002397598873]


def get_precision(dims, bb):
    global weights

    # min. weight[i] / (2 ** (bits[i] - 1))**2
    # s.t. \sum_i bits[i] * dims[i] <= target * \sum_i dims[i]
    weights = np.array(weights)
    dims = np.array(dims, dtype=np.int32)

    print(weights)
    print(dims)

    total_bits = dims.sum() * bb
    L = weights.shape[0]
    var = np.ones([L+1, total_bits+1]) * 1e8
    pred = np.ones([L+1, total_bits+1], dtype=np.int32) * -1
    var[0, 0] = 0
    pred[0, 0] = 0
    for l in range(L):
        for b in range(total_bits):
            if pred[l, b] >= 0:
                for nb in range(1, 9):
                    tb = b + nb * dims[l]
                    if tb <= total_bits:
                        obj = var[l, b] + weights[l]**2 / (2 ** nb - 1)**2
                        if obj < var[l+1, tb]:
                            var[l+1, tb] = obj
                            pred[l+1, tb] = nb
    best_obj = 1e8
    best_tb = -1
    for tb in range(total_bits+1):
        if var[L, tb] <= best_obj:
            best_obj = var[L, tb]
            best_tb = tb

    print('Best obj = ', best_obj)
    bits = np.zeros(L, dtype=np.int32)
    tb = best_tb
    for l in range(L, 0, -1):
        bits[l-1] = pred[l, tb]
        tb -= dims[l-1] * bits[l-1]
        tb = int(tb)

    print(bits)
    return bits
