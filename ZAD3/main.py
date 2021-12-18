import numpy
import math
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

import tools


class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


class HarmonicPlaneWave:
    ''' Класс с уравнением плоской волны для гармонического сигнала в дискретном виде
    A - амплитуда
    Nl - количество ячеек на длину волны.
    phi0 - начальная фаза.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, A, Nl, phi0, Sc=1.0, eps=1.0, mu=1.0):
        self.A = A
        self.Nl = Nl
        self.phi0 = phi0
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return self.A * numpy.sin(
            2 * numpy.pi / self.Nl * (self.Sc * q - numpy.sqrt(self.mu * self.eps) * m) + self.phi0)


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Число Куранта
    Sc = 1.0

    # Время расчета в секундах
    maxTime_s = 80e-9

    # Размер области моделирования в метрах
    maxSize_m = 2.5

    # Дискрет по пространству в м
    dx = 1e-3

    # Скорость обновления графика поля
    speed_refresh = 50

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = 6

    # Магнитная проницаемость
    mu = 1

    # Скорость распространения волны
    v = c / numpy.sqrt(eps)

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / v

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в метрах
    sourcePos_m = maxSize_m / 2
    # Положение источника в отсчетах
    sourcePos = math.floor(sourcePos_m / dx + 0.5)

    # Положение датчика
    probesPos_m = 1
    # Датчики для регистрации поля
    probesPos = [math.floor(probesPos_m / dx + 0.5)]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    probesPos_1 = math.floor(probesPos_m / dx + 0.5)

    # Массивы для Ez и Hy
    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)
    source = HarmonicPlaneWave(0.5, 500, 0, Sc, eps, mu)

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / numpy.sqrt(mu * eps)
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for t in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu) * source.getE(0, t)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps * mu)) *
                          source.getE(-0.5, t + 0.5))
        # Граничные условия для поля E
        Ez[-1] = 0
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # Расчёт спектра сигнала
    EzSpec = fftshift(numpy.abs(fft(probe.E)))

    # Рассчёт шага частоты
    df = 1.0 / (maxTime * dt)
    # Рассчёт частотной сетки
    freq = numpy.arange(-maxTime / 2, maxTime / 2) * df
    # Оформляем сетку
    tlist = numpy.arange(0, maxTime * dt, dt)

    # Вывод сигнала и спектра зарегестрированых в датчике
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlim(0.5e-7, 0.8e-7)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('Ez, В/м')
    ax1.plot(tlist, probe.E)
    ax1.minorticks_on()
    ax1.grid()
    ax2.set_xlim(-2e9, 2e9)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('f, Гц')
    ax2.set_ylabel('|S| / |Smax|, б/р')
    ax2.plot(freq, EzSpec / numpy.max(EzSpec))
    ax2.minorticks_on()
    ax2.grid()
    plt.show()
