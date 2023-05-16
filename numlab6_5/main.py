# var2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

a1 = 1
b1 = 1
a2 = 1
b2 = 0
T = 1


def solution(x, t):
    return (np.exp(x) * np.tan(t) - x) / 2


def f(x, t):
    return np.exp(x) / 2 * (1 / np.cos(t) ** 2 - np.tan(t))


def phi(x):
    return -x / 2


def theta1(t):
    return np.tan(t) - 1 / 2


def theta2(t):
    return (np.exp(1) * np.tan(t) - 1) / 2


def next(curr_t, prev, a, sigma, left, right, n, tau, theta1, theta2):
    x = np.linspace(left, right, n + 1)
    h = (right - left) / n
    prev_t = curr_t - tau

    a0 = a1 - 3 * b1 / (2 * h)
    ai = a ** 2 * sigma * tau / h ** 2
    an = b2 / (2 * h)

    b0 = 4 * b1 / (2 * h)
    bi = -2 * a ** 2 * sigma * tau / h ** 2 - 1
    bn = -4 * b2 / (2 * h)

    c0 = -b1 / (2 * h)
    ci = a ** 2 * sigma * tau / h ** 2
    cn = a2 + 3 * b2 / (2 * h)

    d = [theta1(curr_t)]
    d += [a ** 2 * tau / h ** 2 * (sigma - 1) * (
            prev[i - 1] - 2 * prev[i] + prev[i + 1]) - tau * f(x[i], (
            1 - sigma) * prev_t + sigma * curr_t) - prev[i] for i in range(1, n)]
    d += [theta2(curr_t)]

    b0_ = a0 - c0 * ai / ci
    c0_ = b0 - c0 * bi / ci
    d0_ = d[0] - c0 * d[1] / ci

    an_ = bn - an * bi / ai
    bn_ = cn - an * ci / ai
    dn_ = d[-1] - an * d[-2] / ai

    A = []
    B = []

    A += [d0_ / b0_]
    B += [-c0_ / b0_]

    for i in range(1, n):
        A += [(d[i] - ai * A[i - 1]) / (bi + B[i - 1] * ai)]
        B += [-ci / (bi + B[i - 1] * ai)]

    A += [(dn_ - an_ * A[-1]) / (bn_ + B[-1] * an_)]
    B += [0]

    curr_layer = np.zeros(n + 1)
    curr_layer[-1] = A[-1]

    for i in reversed(range(0, n)):
        curr_layer[i] = A[i] + B[i] * curr_layer[i + 1]

    return curr_layer


def get_error(func, realFunc):
    return max(abs(func[i] - realFunc[i]) for i in range(len(func)))


def solve(a=1, sigma=0.5, left=0, right=1, h=0.01, plot=False):
    n = int((right - left) / h) + 1
    x = np.linspace(left, right, n)
    first = [phi(xx) for xx in x]

    if plot:
        fig, ax = plt.subplots()
        ax.legend(['Численное решение', 'Аналитическое решение'])
        frames = []

    tau = h
    frames_count = int(T / tau)
    t_h = 0

    prev = first
    errors = []

    for i in range(1, frames_count):

        t_h += tau

        real = [solution(xi, t_h) for xi in x]

        next_ = next(t_h, prev, a, sigma, left, right, n - 1, tau, theta1, theta2)

        prev = list(next_)

        error = get_error(next_, real)
        errors.append(error)

        if plot:
            sol, = ax.plot(x, next_, color='g')
            real_sol, = ax.plot(x, real, color='r', linestyle='dotted')
            ax.legend(['Численное решение', 'Аналитическое решение'])
            frames.append([sol, real_sol])

    if plot:
        animation = ArtistAnimation(fig, frames, interval=30, blit=True, repeat=False)
        animation.save("solution.gif", writer='pillow')

    return max(errors)


def plot_errors():
    errs = []
    hs = []
    h = 0.1

    for i in range(10):
        err = solve(h=h, plot=False)
        print(err)
        errs.append(err)
        hs.append(h)
        h /= 2

    plt.loglog(hs, errs, '-o', label='errors')
    coef1 = errs[0] / hs[0]
    y1 = [h * coef1 for h in hs]
    coef2 = errs[0] / hs[0] ** 2
    y2 = [h ** 2 * coef2 for h in hs]
    plt.loglog(hs, y1, ':g', label='loglog y=h')
    plt.loglog(hs, y2, ':r', label='loglog y=h**2')
    plt.title(f'loglog err from step')
    plt.legend()
    plt.show()


plot_errors()
