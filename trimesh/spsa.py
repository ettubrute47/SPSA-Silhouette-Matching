from functools import partial, reduce
from typing import TypedDict, Optional
from collections import ChainMap

import numpy as np


class SPSA_Params(TypedDict):
    a0: float
    c0: float
    A: int
    gamma: float  # 0.101
    alpha: float  # 0.602
    max_delta_theta: float  # normalize
    max_iter: int
    t0: float
    num_approx: int


# a = a0 * (1 + A) ** alpha
# return a / (k + 1 + A) ** alpha


def bernouli_sample(p):
    while True:
        proba = np.random.rand(p)
        values = (proba < 0.5).astype(int) * 2 - 1
        yield values


class OptimSPSA:
    _default_params: SPSA_Params = {
        "gamma": 0.101,
        "alpha": 0.602,
        "t0": 0.5,
        "num_approx": 1,
    }
    _required_params = {"max_iter", "max_delta_theta"}
    _params: SPSA_Params

    def __init__(self, theta_0, loss_fnc, v_fnc=None, blocking=False, **params):
        self._params = dict(ChainMap(params, self._default_params))
        assert (
            len(self._required_params - set(self._params)) == 0
        ), self._required_params
        self.theta_0 = np.asarray(theta_0)
        self.blocking = blocking
        self.loss = loss_fnc
        self.v_fnc = v_fnc
        self._all_loss_history = []
        self._loss_history = []
        self._used_thetas = []
        self._block_history = []
        self._grad_history = []
        self.thetas = [self.theta_0]
        self.k = 0
        self._perturb_gen = bernouli_sample(len(self.theta_0))
        self._c_guess = self._params.get("c0")

    def calibrate(self, debug=False):
        self._params["A"] = self._params["max_iter"] * 0.13
        self._params["c0"] = self._approximate_c(c_guess=self._c_guess)
        # approximate grad, and take magnitude...
        self._params["a0"] = self._approximate_a(self._params["max_delta_theta"])
        if debug:
            print("After calibration: ", self._params)

    def _get_loss(self, theta, v=None):
        loss = self.loss(theta, v=v)
        self._all_loss_history.append(loss)
        self._used_thetas.append(theta)
        return loss

    def _approximate_a(self, max_delta_theta, num_approx=10):
        approx_grad = np.mean([self.approx_grad() for i in range(num_approx)], axis=0)
        # |a / (k + 1 + A) ** alpha * grad| = max_delta_theta
        max_grad = np.abs(approx_grad).max()
        a0 = (
            (1 + self._params["A"]) ** self._params["alpha"]
            * max_delta_theta
            / max_grad
        )
        return a0

    def _approximate_c(self, num_samples=10, c_guess=None):
        losses = [self._get_loss(self.theta_0) for i in range(num_samples)]
        c_est = np.std(losses, ddof=1) * 3 + 1e-10  # over-estimate it...
        if c_guess is None:
            return c_est
        # geometric mean of the guess and estimate
        return np.sqrt(c_est * c_guess)

    def _get_perturb(self):
        return next(self._perturb_gen)

    def approx_grad(self):
        c = self.ck(self.k)
        perturb = self._get_perturb()
        left = self.theta + c * perturb
        right = self.theta - c * perturb
        v = None
        if self.v_fnc is not None:
            v = self.v_fnc()
        diff = self._get_loss(left, v=v) - self._get_loss(right, v=v)

        grad = (diff / (2 * c)) / perturb
        self._grad_history.append(grad)
        return grad

    def ck(self, k):
        return self._params["c0"] / (k + 1) ** self._params["gamma"]

    def ak(self, k):
        return self._params["a0"] / (k + 1 + self._params["A"]) ** self._params["alpha"]

    @property
    def theta(self):
        return self.thetas[-1]

    def step(self):
        a = self.ak(self.k)

        for i in range(self._params["num_approx"]):
            self.approx_grad()
        grad = np.mean(self._grad_history[-self._params["num_approx"] :], axis=0)
        theta_diff = a * grad
        theta_diff_mag = np.sqrt(theta_diff @ theta_diff)
        if theta_diff_mag > self._params["max_delta_theta"]:
            theta_diff *= self._params["max_delta_theta"] / theta_diff_mag

        # we let it be a max...
        loss = self.loss(self.theta - theta_diff)  # don't count it...
        block = False
        if self.blocking and self.k > 5:
            std_loss = np.std(self._loss_history[-10:])
            block = loss >= self._loss_history[-1] + std_loss * self.temp(self.k)
        self._block_history.append(block)
        if not block:
            self.thetas.append(self.theta - theta_diff)
            self._loss_history.append(loss)

        self.k += 1
        return self.theta

    def temp(self, k):
        return self._params["t0"] / (k + 1) ** self._params["gamma"]

    def _print_progress(self):
        print(f"{self.k}: {self._loss_history[-1]}")

    def irun(self, num_steps=None, print_progress=True):
        if num_steps is None:
            num_steps = self._params["max_iter"]
        pi = int(num_steps // 10)
        for i in range(num_steps):
            theta = self.step()
            if print_progress and i % pi == 0:
                self._print_progress()
            yield theta

    def run(self, num_steps=None, print_progress=True):
        for theta in self.irun(num_steps=num_steps, print_progress=print_progress):
            pass
        return theta
