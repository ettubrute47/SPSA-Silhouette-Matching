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
    c_guess: float
    c_std_scale: float
    k0: int
    loss_rng: float


class Box:
    def __init__(self, low: np.ndarray, hi: np.ndarray, size=None):
        self.low = np.asarray(low)
        self.hi = np.asarray(hi)
        self.rng = self.hi - self.low
        self.mag = np.linalg.norm(self.rng)

    def normalize(self, theta: np.ndarray, i=...):
        return (self.clip(theta, i=i) - self.low[i]) / self.rng[i]

    def unnormalize(self, theta_norm: np.ndarray, i=...):
        return self.rng[i] * self.clip(theta_norm, True, i=i) + self.low[i]

    def clip(self, theta: np.ndarray, normalized=False, i=...):
        if normalized:
            return np.clip(theta, 0, 1)
        return np.clip(theta, self.low[i], self.hi[i])


# a = a0 * (1 + A) ** alpha
# return a / (k + 1 + A) ** alpha
class OptimBase:
    _default_params = dict()
    _required_params = set()

    def __init__(
        self,
        theta_0,
        box: Box,
        loss_fnc,
        v_fnc=None,
        verbose=False,
        on_theta_update=None,
        implicit_theta_mask=None,
        **params,
    ):
        self.verbose = verbose
        self._params = ChainMap(params, self._default_params)
        self._params.maps[-1]["max_delta_theta"] = (
            1 / self._params["max_iter"]
        )  # default value for max_delta_theta...
        self.loss = loss_fnc
        self._on_theta_update = on_theta_update
        self.v_fnc = v_fnc
        self.box = box
        self._v = None
        self._implicit_theta_mask = np.ones(len(theta_0), bool)
        if implicit_theta_mask is not None:
            self._implicit_theta_mask = np.array(implicit_theta_mask, bool)
        self._all_loss_history = []
        self._loss_history = []
        self._used_thetas = []
        self._block_history = []
        self._grad_history = []
        self._theta_0 = np.asarray(theta_0)
        self.thetas = [self.box.normalize(self._theta_0)]
        self.k = self._params.get("k0", 0)
        assert (
            len(self._required_params - set(self._params)) == 0
        ), self._required_params

    @property
    def theta_0(self):
        return self.box.normalize(self._theta_0)

    def set_params(self, **params):
        self._params.maps[0].update(params)

    def reset(self, theta_0=None):
        if theta_0 is not None:
            self._theta_0 = np.array(theta_0)
        self._all_loss_history = []
        self._loss_history = []
        self._used_thetas = []
        self._block_history = []
        self._grad_history = []
        self.thetas = [self.theta_0]
        # self.thetas = [np.array(theta_0)]
        self.k = self["k0"]

    def _approximate_a(self, max_delta_theta, num_approx=10):
        approx_grad = np.mean([self.approx_grad() for i in range(num_approx)], axis=0)
        # |a / (k + 1 + A) ** alpha * grad| = max_delta_theta
        max_grad = np.abs(approx_grad).max()
        assert max_grad > 0, f"0 grad for {self.theta_0} normal theta0..."
        a0 = (
            (1 + self._params["A"]) ** self._params["alpha"]
            * max_delta_theta
            / max_grad
        )
        return a0

    def __getitem__(self, key: str):
        return self._params.get(key)

    def _set_param_default(self, key: str, val):
        self._params.maps[-1][key] = val

    def __contains__(self, key: str):
        return key in self._params

    def _is_param_explictly_set(self, key):
        return key in self._params.maps[0]

    def calibrate(self):
        self._set_param_default("A", self["max_iter"] * 0.07)
        # approximate grad, and take magnitude...
        self._set_param_default("a0", self._approximate_a(self["max_delta_theta"]))

    def _get_loss(self, theta):
        raw_theta = self.box.unnormalize(theta)
        loss = self.loss(raw_theta, v=self._v)
        self._all_loss_history.append(loss)
        self._used_thetas.append(theta)
        return loss

    def approx_grad(self):
        pass

    def check_block(self, theta_diff, loss):
        if self["blocking"] and self.k > 5:
            std_loss = np.std(self._loss_history[-10:])
            block = loss >= self._loss_history[-1] + std_loss * self.temp(self.k)
            return block
        return False

    def _update_theta(self, theta_diff):
        if len(self.thetas) > 2 and self._params["momentum"] > 0:
            prev_diff = self.theta - self.thetas[-2]
            alignment = np.dot(
                prev_diff / np.linalg.norm(prev_diff),
                -theta_diff / np.linalg.norm(theta_diff),
            )
            alignment = np.sqrt(max(0, alignment))
            momentum = prev_diff * alignment * self._params["momentum"]
            self.thetas.append(self.theta - theta_diff + momentum)
        else:
            self.thetas.append(self.box.clip(self.theta - theta_diff, True))
        if self._on_theta_update is not None:
            self._on_theta_update(self)

    def step(self):
        a = self.ak
        self._v = None

        for i in range(self._params["num_approx"]):
            self.approx_grad()
        grad = np.mean(self._grad_history[-self._params["num_approx"] :], axis=0)
        theta_diff = a * grad
        theta_diff_mag = np.sqrt(theta_diff @ theta_diff)
        if theta_diff_mag > self._params["max_delta_theta"]:
            theta_diff *= self._params["max_delta_theta"] / theta_diff_mag

        # we let it be a max...
        loss = self._get_loss(self.theta - theta_diff)  # don't count it...
        block = self.check_block(theta_diff, loss)
        self._block_history.append(block)
        if not block:
            self._update_theta(theta_diff)
            self._loss_history.append(loss)

        self.k += 1
        return self.theta

    @property
    def ak(self):
        return self.a_gain(self.k)

    def a_gain(self, k):
        return self._params["a0"] / (k + 1 + self._params["A"]) ** self._params["alpha"]

    def temp(self, k):
        return self._params["t0"] / (k + 1) ** self._params["gamma"]

    def smooth_theta(self):
        """
        V1: just static exponential smooth
        """
        if self["theta_smooth"] > 1 and len(self.thetas) > self["theta_smooth"]:
            return np.average(
                self.thetas[-self["theta_smooth"] :][::-1],
                axis=0,
                weights=(1 / np.arange(1, self["theta_smooth"] + 1)),
            )
        return self.thetas[-1]

    def smooth_theta2(self):
        n = int(self["theta_smooth"] * self.k)  # percent of k
        # also the weights are exponential changing...
        n = int(n * self.perc_k)
        if n <= 1:
            return self.thetas[-1]
        return np.average(
            self.thetas[-n:][::-1],
            axis=0,
            weights=(1 / np.arange(1, n + 1)),
        )

    @property
    def theta(self):
        if self["theta_smooth"] > 0:
            return self.smooth_theta2()
        return self.thetas[-1]

    @property
    def perc_k(self):
        return self.k / self["max_iter"]

    def _print_progress(self):
        print(f"{self.k}: {self._loss_history[-1]}")

    def irun(self, theta_0=None, reset=True, calibrate=True, num_steps=None, **params):
        if theta_0 is not None:
            assert reset, "when you set theta_0 you are also resetting the optimizer"
        self.set_params(**params)
        if reset:
            self.reset(theta_0)
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

    def run(self, theta_0=None, reset=True, calibrate=True, num_steps=None, **params):
        for theta in self.irun(
            theta_0=theta_0,
            reset=reset,
            calibrate=calibrate,
            num_steps=num_steps,
            **params,
        ):
            pass
        return theta

    def rms_dist_from_truth(self, goal_theta, normalized=False):
        assert not normalized
        diffs = self.thetas - self.box.normalize(goal_theta)
        rms_dist = np.sqrt(np.mean(diffs * diffs, axis=1))
        return rms_dist

    def custom_experiment(self, num_trials, setup=None, get_metrics=None):
        """
        get_metrics must return a list of elements that are different types of metrics
        for instance, if you want to collect a list of loss histories only, you'd need to return
        (optim._loss_history, )
        """
        results = []
        for i in range(num_trials):
            if setup is not None:
                setup(self, i)
            self.run()
            if get_metrics is not None:
                run_results = get_metrics(self, i)
            else:
                run_results = (self._loss_history,)
            for i, res in enumerate(run_results):
                if len(results) <= i:
                    results.append([])
                results[i].append(np.array(res))
        for i in range(len(results)):
            results[i] = np.array(results[i]).T
        return results

    def experiment(self, num_trials, theta_history=False, goal_theta=None):
        losses = []
        dists = []
        thetas = []
        for i in range(num_trials):
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


class OptimSA(OptimBase):
    _default_params: SPSA_Params = {
        "gamma": 0.101,
        "alpha": 0.602,
        "t0": 0.5,
        "num_approx": 1,
        "blocking": False,
        "max_iter": 100,
        "momentum": 0.0,
        "c_std_scale": 3.0,
        "k0": 0,
        "theta_smooth": 0,
    }
    _required_params = {"max_iter", "max_delta_theta", "loss_rng"}
    _params: ChainMap | SPSA_Params

    def calibrate(self):
        self._set_param_default("c0", self._approximate_c())
        super().calibrate()
        if self.verbose:
            print("After calibration: ", self._params)

    def _approximate_c(
        self,
        num_samples=10,
    ):
        losses = []
        for i in range(num_samples):
            if self.v_fnc is not None:
                self._v = self.v_fnc()
            losses.append(self._get_loss(self.theta_0))
        # losses = [self._get_loss(self.theta_0) for i in range(num_samples)]
        c_est = (
            np.std(losses, ddof=1) * self["c_std_scale"] / self["loss_rng"] + 1e-10
        )  # over-estimate it...
        if "c_guess" not in self:
            return c_est
        # geometric mean of the guess and estimate
        return np.sqrt(c_est * self["c_guess"])

    @property
    def ck(self):
        return self.c_gain(self.k)

    def c_gain(self, k):
        return self._params["c0"] / (k + 1) ** self._params["gamma"]


class OptimSPSA(OptimSA):
    def __init__(
        self,
        theta_0,
        box: Box,
        loss_fnc,
        v_fnc=None,
        perturb_gen=None,
        verbose=False,
        **params,
    ):
        super().__init__(theta_0, box, loss_fnc, v_fnc, verbose=verbose, **params)
        if perturb_gen is None:
            self._perturb_gen = bernouli_sample(len(self.theta_0))
        else:
            self._perturb_gen = perturb_gen(len(self.theta_0))

    def approx_grad(self):
        c = self.ck
        perturb = next(self._perturb_gen) * self._implicit_theta_mask
        left = self.theta + c * perturb
        right = self.theta - c * perturb
        if self.v_fnc is not None:
            self._v = self.v_fnc()
        diff = self._get_loss(left) - self._get_loss(right)
        grad = np.zeros(self.theta_0.shape)

        grad[self._implicit_theta_mask] = (diff / (2 * c)) / perturb[
            self._implicit_theta_mask
        ]
        # grad = (diff / (2 * c)) / perturb
        self._grad_history.append(grad)
        return grad


class OptimFDSA(OptimSA):
    def approx_grad(self):
        # SPSA
        c = self.ck(self.k)
        P = len(self.theta)
        perturbs = np.eye(P) * c
        perturbs[:, self._implicit_theta_mask == 0] = 0  # I could be wrong dim here...
        if self.v_fnc is not None:
            self._v = self.v_fnc()
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

    theta_0 = np.array([0.4, 1.7])

    def on_theta_update(optim: OptimSA):
        optim.thetas[-1][-1] = optim.box.normalize(
            optim.perc_k * goal_theta[-1] + theta_0[-1] * (1 - optim.perc_k), i=-1
        )

    optim = OptimSPSA(
        theta_0,
        Box(np.zeros(2), np.ones(2) * 10),
        partial(noisy_loss, noise_scale=1e-1),
        verbose=True,
        loss_rng=10,
        implicit_theta_mask=[1, 0],
        # alpha=1,
        max_iter=100,
        # theta_smooth=0.1,
        c_std_scale=1,
        on_theta_update=on_theta_update,
        # momentum=0.95,
    )
    optim.run()

    thetas = optim.box.unnormalize(optim.thetas)
    num_segments = len(thetas)
    for i in range(len(thetas) - 1):
        x1, y1 = thetas[i]
        x2, y2 = thetas[i + 1]
        plt.plot([x1, x2], [y1, y2], color=plt.cm.gray(i / num_segments), marker="x")
    plt.scatter(*theta_0, color="r", marker="x")
    plt.scatter(*goal_theta, color="g", marker="o")
    plt.show()
