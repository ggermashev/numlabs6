# var 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

x_start = 0
x_end = 1
t_max = 1
a = 1

# 2Utt = Uxx - 2 + ( 2t(x-1)**3 - (x-1)t**3 ) / (4 - t**2 * (x-1)**2 ) ** 3/2 f
# u(x,0) = x**2 phi
# Ut(x,0) = (x+1)/2 psi
# u(0,t) = t - arcsin(t/2)  theta
# u(1,t) + 2Ux(1,t) = 5 + 2t mu
# U0 = t + x**2 + arcsin(t(x-1)/2)

alpha = 1
beta = 2


def get_mu(t):
    return 5 + 2 * t


def real_func(x, t):
    return t + x ** 2 + np.arcsin(t * (x - 1) / 2)


def get_f(x, t):
    f = ((2 * t * (x - 1) ** 3 - t ** 3 * (x - 1)) / (4 - t ** 2 * (x - 1) ** 2) ** (3 / 2) - 2)
    return f


def get_psi(x):
    return (x + 1) / 2


def get_phi(x):
    return x ** 2


def get_phi_dif2():
    return 2


def get_theta(t):
    return t - np.arcsin(t / 2)


def right_condition1(mu, u_left, h):
    return (mu + 2 / h * u_left) / (1 + 2 / h)


def right_condition2(mu, u_left, h, u_left_left):
    return (mu - (u_left_left - 4 * u_left) * beta / (2 * h)) / (alpha + 3 * beta / (2 * h))


def start_condition1(origin, x, t, psi):
    res = [0] * len(origin)
    n = len(res)
    for i in range(n):
        res[i] = origin[i] + t * psi(x[i])
    return res


def start_condition2(origin, x, t, psi, a, phi_dif2, f):
    next = np.zeros_like(origin)
    for i in range(len(origin)):
        next[i] = origin[i] + t * psi(x[i]) + t ** 2 / 2 * (a ** 2 * phi_dif2() + f(x[i], 0))
    return next


def get_next_layer(current, prev, a, get_f, h, x, t, current_t, accuracy=1):
    next = [0] * len(current)
    for i in range(1, len(current) - 1):
        next[i] = 2 * current[i] - prev[i] + (
                (t * a / h) ** 2 * (current[i + 1] - 2 * current[i] + current[i - 1]) + t ** 2 * get_f(x[i], current_t)) / 2
    next[0] = get_theta(current_t + t)
    if accuracy == 1:
        next[-1] = right_condition1(get_mu(current_t + t), next[-2], h)
    else:
        next[-1] = right_condition2(get_mu(current_t + t), next[-2], h, next[-3])
    return next


def get_error(func, realFunc):
    return max(abs(func[i] - realFunc[i]) for i in range(len(func)))


def solve(n, c, plot=False, accuracy=1):
    x = np.linspace(x_start, x_end, n)
    h = x[1] - x[0]
    # c = t * a / h
    t = c * h / a
    u0 = get_phi(x)
    current_t = t
    if accuracy == 1:
        u1 = start_condition1(u0, x, t, get_psi)
    else:
        u1 = start_condition2(u0, x, t, get_psi, a, get_phi_dif2, get_f)
    current = u1.copy()
    prev = u0.copy()
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        frames = []
    errors = []

    real = real_func(x, 0)
    if plot:
        line, = ax1.plot(x, u0, color='b', label='calculated')
        realLine, = ax1.plot(x, real, color='g', label='real')
        ax1.legend()
        frames.append([line, realLine])
    error = get_error(u0, real)
    errors.append(error)

    real = real_func(x, t)
    if plot:
        line, = ax1.plot(x, u1, color='b')
        realLine, = ax1.plot(x, real, color='g')
        frames.append([line, realLine])
    error = get_error(u1, real)
    errors.append(error)

    while current_t < t_max:
        next = get_next_layer(current, prev, a, get_f, h, x, t, current_t, accuracy=accuracy)
        current_t += t
        real = real_func(x, current_t)

        if plot:
            line, = ax1.plot(x, next, color='b')
            realLine, = ax1.plot(x, real, color='g')
            frames.append([line, realLine])
        error = get_error(next, real)
        errors.append(error)

        prev = current.copy()
        current = next.copy()

    if plot:
        animation = ArtistAnimation(
            fig,  # фигура, где отображается анимация
            frames,  # кадры
            interval=30,  # задержка между кадрами в мс
            blit=True,  # использовать ли двойную буферизацию
            repeat=False)  # зацикливать ли анимацию

        animation.save(f"solution_acc{accuracy}.gif", writer='pillow')

    return max(errors)


def plot_errors(accuracy=1):
    errs = []
    hs = []
    for n in range(100, 400, 20):
        err = solve(n, 0.7, plot=False, accuracy=accuracy)
        print(err)
        errs.append(err)
        hs.append((x_end - x_start) / n)
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


# solve(200, 0.7, True, accuracy=1)
# solve(200, 0.7, True, accuracy=2)
plot_errors(1)
plot_errors(2)
