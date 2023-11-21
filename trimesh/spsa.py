from functools import partial, reduce
from itertools import chain, combinations
from typing import TypedDict, Optional
from collections import ChainMap

import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def balanced_bernouli(P):
    ps = list(powerset(range(P)))
    while True:
        order = np.random.choice(len(ps), len(ps), False)
        for i in order:
            s = np.ones(P)
            s[list(ps[i])] = -1
            yield s


def bernouli_sample(p):
    while True:
        proba = np.random.rand(p)
        values = (proba < 0.5).astype(int) * 2 - 1
        yield values


def segmented_uniform_sample(p, width=1.0):
    while True:
        proba = np.random.rand(p)

        values = (proba < 0.5).astype(int) * 2 - 1  # if proba < 0.5, value is 1

        yield (proba + 0.5) * width * values


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
    momentum: float


# a = a0 * (1 + A) ** alpha
# return a / (k + 1 + A) ** alpha
class OptimBase:
    def __init__(self, theta_0, loss_fnc, v_fnc=None, verbose=False, **params):
        self.verbose = verbose
        self.theta_0 = np.asarray(theta_0)
        self._params = params
        self.loss = loss_fnc
        self.v_fnc = v_fnc
        self._all_loss_history = []
        self._loss_history = []
        self._used_thetas = []
        self._block_history = []
        self._grad_history = []
        self.thetas = [self.theta_0]
        self.k = 0

    def reset(self):
        self._all_loss_history = []
        self._loss_history = []
        self._used_thetas = []
        self._block_history = []
        self._grad_history = []
        self.thetas = [self.theta_0]
        self.k = 0

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

    def calibrate(self):
        self._params["A"] = self._params["max_iter"] * 0.13
        # approximate grad, and take magnitude...
        self._params["a0"] = self._approximate_a(self._params["max_delta_theta"])

    def _get_loss(self, theta, v=None):
        loss = self.loss(theta, v=v)
        self._all_loss_history.append(loss)
        self._used_thetas.append(theta)
        return loss

    def approx_grad(self):
        pass

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
        if self._params["blocking"] and self.k > 5:
            std_loss = np.std(self._loss_history[-10:])
            block = loss >= self._loss_history[-1] + std_loss * self.temp(self.k)
        self._block_history.append(block)
        if not block:
            if len(self.thetas) > 2:
                prev_diff = self.theta - self.thetas[-2]
                alignment = np.dot(
                    prev_diff / np.linalg.norm(prev_diff),
                    -theta_diff / np.linalg.norm(theta_diff),
                )
                alignment = np.sqrt(max(0, alignment))
                momentum = prev_diff * alignment * self._params["momentum"]
                self.thetas.append(self.theta - theta_diff + momentum)
            else:
                self.thetas.append(self.theta - theta_diff)
            self._loss_history.append(loss)

        self.k += 1
        return self.theta

    def ak(self, k):
        return self._params["a0"] / (k + 1 + self._params["A"]) ** self._params["alpha"]

    def temp(self, k):
        return self._params["t0"] / (k + 1) ** self._params["gamma"]

    @property
    def theta(self):
        return self.thetas[-1]

    def _print_progress(self):
        print(f"{self.k}: {self._loss_history[-1]}")

    def irun(self, reset=True, calibrate=True, num_steps=None):
        if reset:
            self.reset()
        if calibrate:
            self.calibrate()
        if num_steps is None:
            num_steps = self._params["max_iter"]
        pi = int(num_steps // 10)
        for i in range(num_steps):
            theta = self.step()
            if self.verbose and i % pi == 0:
                self._print_progress()
            yield theta

    def run(self, reset=True, calibrate=True, num_steps=None):
        for theta in self.irun(reset=reset, calibrate=calibrate, num_steps=num_steps):
            pass
        return theta

    def rms_dist_from_truth(self, goal_theta):
        diffs = self.thetas - goal_theta
        rms_dist = np.sqrt(np.mean(diffs * diffs, axis=1))
        return rms_dist

    def experiment(self, num_trials, theta_history=False, goal_theta=None):
        losses = []
        dists = []
        thetas = []
        for i in range(num_trials):
            self.reset()
            self.calibrate()
            self.run()
            losses.append(self._loss_history.copy())
            if goal_theta is not None:
                dists.append(self.rms_dist_from_truth(goal_theta))
            if theta_history:
                thetas.append(self.thetas.copy())
        results = [np.array(losses).T]
        if theta_history:
            results.append(np.array(thetas).T)
        if goal_theta is not None:
            results.append(np.array(dists).T)
        return results


class OptimSPSA(OptimBase):
    _default_params: SPSA_Params = {
        "gamma": 0.101,
        "alpha": 0.602,
        "t0": 0.5,
        "num_approx": 1,
        "blocking": False,
        "max_iter": 100,
        "momentum": 0.0,
    }
    _required_params = {"max_iter", "max_delta_theta"}
    _params: SPSA_Params

    def __init__(
        self,
        theta_0,
        loss_fnc,
        v_fnc=None,
        perturb_gen=None,
        verbose=False,
        **params,
    ):
        params = ChainMap(params, self._default_params)
        assert len(self._required_params - set(params)) == 0, self._required_params
        super().__init__(theta_0, loss_fnc, v_fnc, verbose=verbose, **params)
        if perturb_gen is None:
            self._perturb_gen = bernouli_sample(len(self.theta_0))
        else:
            self._perturb_gen = perturb_gen(len(self.theta_0))
        self._c_guess = self._params.get("c0")

    def calibrate(self):
        self._params["c0"] = self._approximate_c(c_guess=self._c_guess)
        super().calibrate()
        if self.verbose:
            print("After calibration: ", self._params)

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


class OptimFDSA(OptimBase):
    _default_params: SPSA_Params = {
        "gamma": 0.101,
        "alpha": 0.602,
        "t0": 0.5,
        "num_approx": 1,
        "blocking": False,
        "max_iter": 100,
        "momentum": 0.0,
    }
    _required_params = {"max_iter", "max_delta_theta"}
    _params: SPSA_Params

    def __init__(
        self,
        theta_0,
        loss_fnc,
        v_fnc=None,
        perturb_gen=None,
        verbose=False,
        **params,
    ):
        params = ChainMap(params, self._default_params)
        assert len(self._required_params - set(params)) == 0, self._required_params
        super().__init__(theta_0, loss_fnc, v_fnc, verbose=verbose, **params)
        if perturb_gen is None:
            self._perturb_gen = bernouli_sample(len(self.theta_0))
        else:
            self._perturb_gen = perturb_gen(len(self.theta_0))
        self._c_guess = self._params.get("c0")

    def calibrate(self):
        self._params["c0"] = self._approximate_c(c_guess=self._c_guess)
        super().calibrate()
        if self.verbose:
            print("After calibration: ", self._params)

    def _approximate_c(self, num_samples=10, c_guess=None):
        losses = [self._get_loss(self.theta_0) for i in range(num_samples)]
        c_est = np.std(losses, ddof=1) * 3 + 1e-10  # over-estimate it...
        if c_guess is None:
            return c_est
        # geometric mean of the guess and estimate
        return np.sqrt(c_est * c_guess)

    def ck(self, k):
        return self._params["c0"] / (k + 1) ** self._params["gamma"]

    def approx_grad(self):
        # SPSA
        c = self.ck(self.k)
        P = len(self.theta)
        perturbs = np.eye(P) * c
        left_thetas = self.theta + perturbs
        right_thetas = self.theta - perturbs
        left_loss = np.apply_along_axis(self._get_loss, arr=left_thetas, axis=1)
        right_loss = np.apply_along_axis(self._get_loss, arr=right_thetas, axis=1)
        grad = (left_loss - right_loss) / (2 * c)
        self._grad_history.append(grad)
        return grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    goal_theta = np.array([1, 1])

    def noise_source(noise_scale=1e-3):
        return np.random.normal(0, noise_scale, size=len(goal_theta))

    def noisy_loss(theta, noise_scale=1e-3, v=None):
        if v is None:
            v = noise_source(noise_scale)
        diff = np.asarray(theta) - goal_theta + v
        return np.sqrt(diff @ diff)

    theta_0 = np.array([-1.5, -1.5])

    optim = OptimSPSA(
        theta_0,
        partial(noisy_loss, noise_scale=1e-2),
        verbose=True,
        max_delta_theta=0.15,
        # alpha=1,
        max_iter=100,
        # momentum=0.95,
    )
    optim.run()

    thetas = optim.thetas
    num_segments = len(thetas)
    for i in range(len(thetas) - 1):
        x1, y1 = thetas[i]
        x2, y2 = thetas[i + 1]
        plt.plot([x1, x2], [y1, y2], color=plt.cm.gray(i / num_segments), marker="x")
    plt.scatter(*theta_0, color="r", marker="x")
    plt.scatter(*goal_theta, color="g", marker="o")
    plt.show()
