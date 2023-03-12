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


a = 1
x_min = 0.0
x_max = 4.5
t_max = (x_max - 1.5) / a
c = 0.7

x0 = 0.5
t0 = 0.8
e = 0.45
n = 500


def solve(func):
    x = np.linspace(x_min, x_max, n)
    h = x[1] - x[0]
    e = 0.3
    u = [func(point, x0, e) for point in x]

    fig, ax = plt.subplots()

    frames = []
    line, = ax.plot(x, u, color='b')
    realLine, = ax.plot(x, u, color='g')
    frames.append([line, realLine])

    errors = []
    t = 0
    k = 0

    while t < t_max:
        t += c * h / a
        v = [0 if j == 0 or j == len(x) - 1 else u[j] + c ** 2 / 2 * (u[j + 1] - 2 * u[j] + u[j - 1]) - c / 2 * (
                    u[j + 1] - u[j - 1]) for j
             in range(len(x))]
        realFunc = [func(point - a * t, x0, e) for point in x]
        line, = ax.plot(x, v, color='b')
        realLine, = ax.plot(x, realFunc, color='g')
        frames.append([line, realLine])
        errors.append(get_error(v, realFunc))
        u, v = v, u
        k += 1

    animation = ArtistAnimation(
        fig,  # фигура, где отображается анимация
        frames,  # кадры
        interval=30,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)  # зацикливать ли анимацию

    animation.save(f'{func.__name__}.gif', writer='imagemagick')


solve(exp_func)
