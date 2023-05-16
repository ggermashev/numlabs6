import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def step_func(x, x0, eps):
    xi = np.abs((x - x0) / eps)
    return np.heaviside(1.0 - xi, 0.0)


def parabola(x, x0, eps):
    xi = np.abs((x - x0) / eps)
    return (1.0 - xi ** 2) * np.heaviside(1.0 - xi, 0.0)


def exp_func(x, x0, eps):
    xi = np.abs((x - x0) / eps)
    return np.exp(-xi ** 2 / np.abs(1.0 - xi ** 2)) * np.heaviside(1.0 - xi, 0.0)


def sin_func(x, x0, eps):
    xi = np.abs((x - x0) / eps)
    return np.cos(0.5 * np.pi * xi) ** 3 * np.heaviside(1.0 - xi, 0.0)


def get_error(func, realFunc):
    return max(abs(func[i] - realFunc[i]) for i in range(len(func)))


def limiter(r):
    return (r + np.abs(r)) / (1 + np.abs(r))


a = 1
x_min = 0.0
x_max = 4.5
t_max = (x_max - 1.5) / a
c = 0.7

x0 = 0.5
t0 = 0.8
e = 0.3


def solve(func, n, color, ax1=None, ax2=None, frames=None, fig=None, flag=True):
    x = np.linspace(x_min, x_max, n)
    h = x[1] - x[0]
    u = [func(point, x0, e) for point in x]

    if ax1 and ax2 and frames and fig:
        line, = ax1.plot(x, u, color='b')
        realLine, = ax1.plot(x, u, color='g')
        frames.append([line, realLine])

    errors = []
    t = 0
    k = 0
    while t < t_max:
        t += c * h / a
        v = np.zeros_like(u)
        v[0] = 0
        v[1] = 0.5 * c ** 2 * (u[2] - 2 * u[1] + u[0]) - 0.5 * c * (u[2] - u[0]) + u[1]
        for i in range(2, len(u) - 1):
            r1 = (u[i-1] - u[i - 2]) / (u[i] - u[i-1]) if u[i] != u[i - 1] else 1
            r2 = (u[i] - u[i - 1]) / (u[i + 1] - u[i]) if u[i + 1] != u[i] else 1
            f_up1 = u[i-1]
            f_up2 = u[i]
            f_lax1 = (u[i-1] * (1 + c) / 2 + u[i] * (1 - c) / 2)
            f_lax2 = (u[i] * (1 + c) / 2 + u[i+1] * (1 - c) / 2)
            phi1 = limiter(r1)
            phi2 = limiter(r2)
            f1 = f_up1 + phi1 * (f_lax1 - f_up1)
            f2 = f_up2 + phi2 * (f_lax2 - f_up2)
            v[i] = u[i] - c * (f2 - f1)
        # print(v)
        realFunc = [func(point - a * t, x0, e) for point in x]
        error = get_error(v, realFunc)
        errors.append(error)
        if ax1 and ax2 and frames and fig:
            if flag:
                line, = ax1.plot(x, v, color=f'{color}', label=f'n={n}')
                realLine, = ax1.plot(x, realFunc, color='y', label='real')
                err_plot, = ax2.plot(x[:len(errors)], errors, color=f'{color}', label=f'n={n}')
                flag = not flag
            else:
                line, = ax1.plot(x, v, color=f'{color}')
                realLine, = ax1.plot(x, realFunc, color='y')
                err_plot, = ax2.plot(x[:len(errors)], errors, color=f'{color}')
            frames.append([line, realLine, err_plot])
        u, v = v, u
        k += 1
    return max(errors)


def plot(func):
    n1 = 200
    fig, (ax1, ax2) = plt.subplots(1, 2)
    frames1 = []
    solve(func, n1, 'b', ax1, ax2, frames1, fig)
    ax1.legend()
    ax2.legend()
    animation = ArtistAnimation(
        fig,  # фигура, где отображается анимация
        frames1,  # кадры
        interval=30,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)  # зацикливать ли анимацию

    animation.save(f'{func.__name__}.gif', writer='imagemagick')


def plot_errors(func):
    errs = []
    hs = []
    for n in range(400, 800, 20):
        err = solve(func, n, 'b')
        print(err)
        errs.append(err)
        hs.append((x_max - x_min) / n)
    plt.loglog(hs, errs, '-o', label='errors')
    coef1 = errs[0] / hs[0]
    y1 = [h * coef1 for h in hs]
    coef2 = errs[0] / hs[0] ** 2
    y2 = [h ** 2 * coef2 for h in hs]
    plt.loglog(hs, y1, ':g', label='loglog y=h')
    plt.loglog(hs, y2, ':r', label='loglog y=h**2')
    plt.title(f'loglog err from step {func.__name__}')
    plt.legend()
    plt.show()


# plot(sin_func)
# plot(step_func)
# plot(parabola)
# plot(exp_func)

plot_errors(sin_func)
plot_errors(step_func)
plot_errors(parabola)
plot_errors(exp_func)

