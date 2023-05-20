def logloss(act, preds):
    epsilon = 1e-15
    preds = sp.maximum(epsilon, preds)
    preds = sp.minimum(1 - epsilon, preds)
    ll = sum(act * sp.log(preds) + sp.subtract(1, act) * sp.log(sp.subtract(1, preds)))
    ll = ll * -1.0 / len(act)
    return ll