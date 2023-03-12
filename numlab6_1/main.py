import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

h = 0.025
n = 500
c = 0.7
start = 0
end = 10
initStart = 1
initEnd = 2
a = 1
#c = at/h
t_end = 7

def exp_func(x, x_0, e):
    if abs(x - x_0) >= e:
        return 0
    return np.exp(-(x - x_0) ** 2 / (e ** 2 - (x - x_0) ** 2))


def get_real_sqr(t, x):
    # if (initStart + i * c * h <= x <= initEnd + i * c * h):
    if initStart + a * t <= x <= initEnd + a * t:
        return 1
    return 0


def get_real_exp(t, x, x_0, e):
    # if abs(x - x_0 - i * c * h) >= e:
    if abs(x - x_0 - a * t) >= e:
        return 0
    return np.exp(-(x - x_0 - a * t) ** 2 / (e ** 2 - (x - x_0 - a * t) ** 2))


def get_error(func, realFunc):
    return max(abs(func[i] - realFunc[i]) for i in range(len(func)))


def plot_angle_sqr():
    # x = np.arange(start, end, h)
    x = np.linspace(start, end, n)
    h = (end - start) / n
    u = [1 if (initStart <= point <= initEnd) else 0 for point in x]

    fig, ax = plt.subplots()

    frames = []
    frames.append(ax.plot(x, u, color='b'))
    frames.append(ax.plot(x, [get_real_sqr(0, point) for point in x], color='g'))

    errors = []
    t = 0
    k = 0
    # for k in range(len(x)):
    while t < t_end:
        t += c * h / a
        v = [0 if j == 0 else u[j] - c * (u[j] - u[j - 1]) for j in range(len(x))]
        realFunc = [get_real_sqr(t, point) for point in x]
        line, = ax.plot(x, v, color='b')
        realLine, = ax.plot(x, realFunc, color='g')
        frames.append([line, realLine])
        errors.append(get_error(v, realFunc))
        u, v = v, u
        k += 1

    print(errors[-1])
    animation = ArtistAnimation(
        fig,  # фигура, где отображается анимация
        frames,  # кадры
        interval=30,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)  # зацикливать ли анимацию

    animation.save('angle_sqr.gif', writer='imagemagick')


def plot_angle_exp():
    nn = np.arange(1000, 5000, 500)
    x = np.linspace(start, end, n)
    # h = (end - start) / n
    h = x[1] - x[0]
    x_0 = 1
    e = 0.3
    u = [exp_func(point, x_0, e) for point in x]

    fig, ax = plt.subplots()

    frames = []
    line, = ax.plot(x, u, color='b')
    func = [get_real_exp(0, point, x_0, e) for point in x]
    realLine, = ax.plot(x, func, color='g')
    frames.append([line, realLine])

    errors = []
    t = 0
    k = 0

    while t < t_end:
        t += c * h / a
        v = [0 if j == 0 else u[j] - c * (u[j] - u[j - 1]) for j in range(len(x))]
        realFunc = [get_real_exp(t, point, x_0, e) for point in x]
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

    animation.save('angle_exp.gif', writer='imagemagick')

def calc_errors():
    nn = np.arange(3000, 7000, 500)
    hh = []
    all_errs = []
    for n in range(3000,7000, 500):
        x = np.linspace(start, end, n)
        # h = (end - start) / n
        h = x[1] - x[0]
        hh.append(h)
        x_0 = 1
        e = 0.3
        u = [exp_func(point, x_0, e) for point in x]
        errors = []
        t = 0
        k = 0
        while t < t_end:
            t += c * h / a
            v = [0 if j == 0 else u[j] - c * (u[j] - u[j - 1]) for j in range(len(x))]
            realFunc = [get_real_exp(t, point, x_0, e) for point in x]
            errors.append(get_error(v, realFunc))
            u, v = v, u
            k += 1
        print(len(errors))
        print(max(errors))
        all_errs.append(max(errors))
    plt.plot(hh, all_errs)
    plt.show()

def plot_quatro_sqr():
    x = np.arange(start, end, h)
    u = [1 if (initStart <= point <= initEnd) else 0 for point in x]

    fig, ax = plt.subplots()

    frames = []
    frames.append(ax.plot(x, u, color='b'))

    for k in range(len(x)):
        v = [0 if (j == 0 or j == len(x) - 1) else u[j] - c / 2 * (u[j + 1] - u[j - 1]) for j in range(len(x))]
        line, = ax.plot(x, v, color='b')
        frames.append([line])
        u, v = v, u

    animation = ArtistAnimation(
        fig,  # фигура, где отображается анимация
        frames,  # кадры
        interval=30,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)  # зацикливать ли анимацию

    animation.save('quatro_sqr.gif', writer='imagemagick')


calc_errors()
