# %%
from itertools import repeat
import numpy as np


def bernouli_sample(p):
    while True:
        proba = np.random.rand(p)
        values = (proba < 0.5).astype(int) * 2 - 1
        yield values


def a_gain(a0, A, alpha=0.602):
    """Rule of thumb, A ~ <=10% max iterations expected
    a0 to be smallest of desired changes among theta early on"""
    k = 0
    a = a0 * (1 + A) ** alpha
    while True:
        kc = yield a / (k + 1 + A) ** alpha
        k += 1
        if kc is not None:
            print(f"Setting k = {kc}")
            k = kc
            yield  # pause


def c_gain(c, gamma=0.101):
    """Rule of thumb, c ~ std loss noise (e in y = L + e)"""
    k = 0
    while True:
        kc = yield c / (k + 1) ** gamma
        k += 1
        if kc is not None:
            print(f"Setting k = {kc}")
            k = kc
            yield  # pause


def approx_gradient(loss, perturb, theta, c):
    # SPSA
    diff = loss(theta + c * perturb) - loss(theta - c * perturb)
    return (diff / (2 * c)) / perturb


def optim_spsa(theta_0, loss_fnc, ak_gen, ck_gen, max_theta_diff=0.05):
    theta = theta_0
    P = len(theta)
    for perturb, ak, ck in zip(bernouli_sample(P), ak_gen, ck_gen):
        grad = approx_gradient(loss_fnc, perturb, theta, ck)
        theta_diff = ak * grad
        theta_diff_mag = np.sqrt(theta_diff @ theta_diff)
        if theta_diff_mag > max_theta_diff:
            theta_diff *= max_theta_diff / theta_diff_mag
        theta = theta - theta_diff
        sig = yield theta
        if sig is not None:
            theta = sig  # theta_0
            ak.send(0)
            ck.send(0)  # reset
            yield


def optim_newtons(theta_0, loss_fnc, grad_fnc, hess_fnc, ak_gen, min_ak=1e-12):
    theta = theta_0
    ak_mult = 1.0
    last_loss = loss_fnc(theta_0)
    if isinstance(ak_gen, float):
        ak_gen = repeat(ak_gen)
    while True:
        grad = grad_fnc(theta)
        hess = hess_fnc(theta)
        ht = grad / hess
        ak = next(ak_gen) * ak_mult
        if ak < min_ak:
            print("Reached min ak")
            break
        theta_c = theta - ak * ht
        loss = loss_fnc(theta_c)
        if last_loss < loss:
            ak_mult *= 0.5
            continue
        ak_mult = 1.0
        last_loss = loss
        theta = theta_c
        sig = yield theta
        if sig is not None:
            # sig is theta_0
            theta = sig
            ak_gen.send(0)  # reset ak schedule
            yield  # pause


if __name__ == "__main__":

    def loss_7p14(theta):
        return theta @ np.arange(1, len(theta) + 1) + np.prod(1 / theta)

    def grad_7p14(theta):
        # prod tj^-1 either way
        prod_part = np.prod(1 / theta)
        return np.arange(1, len(theta) + 1) - prod_part / theta

    def hess_7p14(theta):
        prod_part = np.prod(1 / theta)
        return 2 * prod_part / (theta**2)

    optim = optim_newtons(np.ones(10), loss_7p14, grad_7p14, hess_7p14, 0.1)
    for i, theta in zip(range(100), optim):
        if i % 10 == 0:
            print(f"Iter {i}: {loss_7p14(theta)}")
            print(
                "   Theta:",
                " ".join([np.format_float_positional(ti, 3) for ti in theta]),
            )

    print(f"Iter {i}: {loss_7p14(theta)}")
    print("   Theta:", " ".join([np.format_float_positional(ti, 3) for ti in theta]))

    optim = optim_spsa(np.ones(10), loss_7p14, a_gain(0.001, 100), c_gain(0.015))
    for i, theta in zip(range(100), optim):
        if i % 10 == 0:
            print(f"Iter {i}: {loss_7p14(theta)}")
            print(
                "   Theta:",
                " ".join([np.format_float_positional(ti, 3) for ti in theta]),
            )

# %%
