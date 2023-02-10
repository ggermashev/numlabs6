import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

h = 0.025
c = 0.8
start = 0
end = 10
initStart = 1
initEnd = 2

def exp_func(x,x_0,e):
    if abs(x - x_0) >= e:
        return 0
    return np.exp(-(x-x_0)**2/(e**2 - (x-x_0)**2))


def plot_angle_sqr():
    x = np.arange(start, end, h)
    u = [1 if (initStart <= point <= initEnd) else 0 for point in x]
    current = 0

    fig, ax = plt.subplots()

    frames = []
    frames.append(ax.plot(x, u, color='b'))


    for k in range(len(x)):
        v = [0 if j == 0 else u[j] - c * (u[j] - u[j - 1]) for j in range(len(x))]
        line, = ax.plot(x, v, color='b')
        frames.append([line])
        u, v = v, u
        current += 1


    animation = ArtistAnimation(
        fig,                # фигура, где отображается анимация
        frames,              # кадры
        interval=30,        # задержка между кадрами в мс
        blit=True,          # использовать ли двойную буферизацию
        repeat=False)       # зацикливать ли анимацию

    animation.save('angle_sqr.gif', writer='imagemagick')


def plot_angle_exp():
    x = np.arange(start, end, h)
    x_0 = 1
    e = 0.3
    u = [exp_func(point, x_0, e) for point in x]
    current = 0

    fig, ax = plt.subplots()

    frames = []
    frames.append(ax.plot(x, u, color='b'))

    for k in range(len(x)):
        v = [0 if j == 0 else u[j] - c * (u[j] - u[j - 1]) for j in range(len(x))]
        line, = ax.plot(x, v, color='b')
        frames.append([line])
        u, v = v, u
        current += 1

    animation = ArtistAnimation(
        fig,  # фигура, где отображается анимация
        frames,  # кадры
        interval=30,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)  # зацикливать ли анимацию

    animation.save('angle_exp.gif', writer='imagemagick')


def plot_quatro_sqr():
    x = np.arange(start, end, h)
    u = [1 if (initStart <= point <= initEnd) else 0 for point in x]
    current = 0

    fig, ax = plt.subplots()

    frames = []
    frames.append(ax.plot(x, u, color='b'))

    for k in range(len(x)):
        v = [0 if (j == 0 or j == len(x) - 1) else u[j] - c / 2 * (u[j + 1] - u[j - 1]) for j in range(len(x))]
        line, = ax.plot(x, v, color='b')
        frames.append([line])
        u, v = v, u
        current += 1

    animation = ArtistAnimation(
        fig,                # фигура, где отображается анимация
        frames,              # кадры
        interval=30,        # задержка между кадрами в мс
        blit=True,          # использовать ли двойную буферизацию
        repeat=False)       # зацикливать ли анимацию

    animation.save('quatro_sqr.gif', writer='imagemagick')


plot_angle_sqr()
plot_angle_exp()

