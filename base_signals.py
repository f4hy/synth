import logging
import numpy as np
PI = np.pi
SAMPLERATE = 44100


def sine_wave(frequency=440.0, amplitude=0.5):
    logging.info("Creating sine wave signal of {}".format(frequency))
    t = np.arange(0, 2*PI, 2*PI/SAMPLERATE)
    s = amplitude*np.sin(t*frequency)
    return s

sine = sine_wave

def square(frequency=440.0):
    logging.info("Creating square wave signal of {}".format(frequency))
    t = np.arange(0, 2*PI, 2*PI/SAMPLERATE)

    def minus_or_one(x):
        if x < 0:
            return -1
        else:
            return 1
    vfun = np.vectorize(minus_or_one)
    s = vfun(np.sin(t*frequency))
    return s


def harmonics(frequency=440.0, number=50):
    logging.info("Creating signal and the first {} harmonics of {}".format(number, frequency))
    amp = 1.0
    decay = 0.6
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1, number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)
    return signal


def equal_energy(frequency=440.0, number=50):
    """energy is propto (frequency)^2(amp)^2 so if we have (n*frequency)
    to keep the amplitude the same we have to have (amp)^2/(n^2)=(amp/n)^2
    """

    logging.info("Creating signal and the first {} harmonics of {}".format(number, frequency))
    amp = 1.0
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1, number):
        nth_amp = amp/((n+1))
        nth_frequncy += frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)
    return signal


def piano(frequency=440.0):
    amp = 1.0
    signal = sine_wave(frequency=frequency, amplitude=amp)

    # took this from a "piano" profile
    ff = {1: 0.8, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1, 6: 0.2, 7: 0.1, 8: 0.01, 9: 0.02, 12: 0.02}

    nth_frequncy = frequency
    for harmonic, amp in ff.items():
        nth_frequncy = frequency+harmonic*frequency
        sine = sine_wave(frequency=nth_frequncy, amplitude=amp)
        signal = signal+sine
    return signal


def even_harmonics(frequency=440.0, number=50):
    logging.info("Creating signal and the first {} even harmonics of {}".format(number, frequency))
    amp = 1.0
    decay = 0.8
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1, number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += 2*frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)
    return signal
