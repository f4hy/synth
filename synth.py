#!/usr/bin/env python3
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyaudio

PI = np.pi

SAMPLERATE = 44100


def signal_plot(full_axe, partial_axe, signal, fundamental):
    full_axe.set_title("Full Signal")
    full_axe.plot(signal)
    full_axe.set_ylim(-1.1, 1.1)

    partial_axe.set_title("Signal 2 cycles")
    partial_axe.plot(signal[0:int(fundamental*2)])
    partial_axe.set_ylim(-1.1, 1.1)
    plt.draw()


def normalize(signal):
    factor = max(signal)
    return signal / factor


def harmonics(frequency=440.0, number=50, norm=True):
    logging.info("Creating signal and the first {} harmonics of {}".format(number, frequency))
    amp = 1.0
    decay = 0.6
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1, number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)

    if norm:
        return normalize(signal)
    else:
        return signal


def custom(frequency=440.0, norm=True):
    amp = 1.0
    signal = sine_wave(frequency=frequency, amplitude=amp)

    # took this from a "piano" profile doesn't look like it at all
    ff = {1: 0.8, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1, 6: 0.2, 7: 0.1, 8: 0.01, 9: 0.02, 12: 0.02}

    nth_frequncy = frequency
    for harmonic, amp in ff.items():
        nth_frequncy = frequency+harmonic*frequency
        sine = (harmonic**(0.1))*sine_wave(frequency=nth_frequncy, amplitude=amp)
        l = len(sine)
        d = attenuate(l/2+l/(harmonic**(2)), range(l))
        signal = signal+d*sine

    if norm:
        return normalize(signal)
    else:
        return signal


def even_harmonics(frequency=440.0, number=50, norm=True):
    logging.info("Creating signal and the first {} even harmonics of {}".format(number, frequency))
    amp = 1.0
    decay = 0.8
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1, number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += 2*frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)

    if norm:
        return normalize(signal)
    else:
        return signal


def gaussian(x, mu, sigma):
    gaussian = np.exp(-np.power(x-mu, 2.) / (2.0*sigma*sigma))
    return gaussian


def attenuate(sigma, steps):
    mu = 0
    return np.array([np.exp(-np.power(x-mu, 2.) / (2.0*sigma*sigma)) for x in steps])


def envelope(signal):
    """Put a gaussian envelope over the signal"""
    l = len(signal)
    mu = len(signal)/2
    x = range(l)
    y = np.array([0.8*gaussian(i, mu/3, mu/2) for i in x])
    return (signal*y)


def sine_wave(frequency=440.0, amplitude=0.5):
    logging.info("Creating sine wave signal of {}".format(frequency))
    t = np.arange(0, 2*PI, 2*PI/SAMPLERATE)
    s = amplitude*np.sin(t*frequency)
    return s


def square_wave(frequency=440.0):
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


def play(rawdata):
    data = rawdata.astype(np.float32).tostring()

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels = 1,
                    rate = SAMPLERATE,
                    output = True)
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

def fft(signal, axe=None):
    sp = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.shape[-1], d=1.0/SAMPLERATE)
    spectrum = np.abs(sp)

    if axe:
        axe.set_title("FFT")
        axe.set_xlabel("Hz")
        axe.plot(freq, spectrum)
        plt.draw()

    peak = max(spectrum)
    fundamental = freq[list(spectrum).index(peak)]
    logging.info("Fundamental is {}".format(fundamental))
    return fundamental


def synth(options):
    """ Generate sound from sin waves """
    logging.debug("Called with {}".format(options))

    plot = not options.dont_plot

    if options.signal == "harmonics":
        signal = harmonics(number=50, frequency=440.0)
    if options.signal == "even-harmonics":
        signal = even_harmonics(number=500, frequency=440.0)
    if options.signal == "square":
        signal = square_wave()
    if options.signal == "sine":
        signal = sine_wave()
    if options.signal == "custom":
        signal = custom()

    if options.guassian:
        signal = envelope(signal)

    if plot:
        fig, plots = plt.subplots(3)
        full_signal_axe, partial_signal_axe, fft_axe = plots
    else:
        full_signal_axe, partial_signal_axe, fft_axe = None

    fundamental = fft(signal, fft_axe)
    play(signal)

    if(options.output_stub):
        logging.warn("Saving audio file not supported YET!")

    if plot:
        signal_plot(full_signal_axe, partial_signal_axe, signal, fundamental)
        if(options.output_stub):
            fig.set_size_inches(18.5, 18.5)
            ofilename = options.output_stub+".png"
            logging.info("Saving plot to {}".format(ofilename))
            plt.savefig(ofilename, dpi=400)
            return
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound from sin waves")
    signal_choices = ["sine", "square", "harmonics", "even-harmonics", "custom"]
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-s", "--signal", required=True, choices=signal_choices,
                        help="which signal to generate", default="harmonics")
    parser.add_argument("-dp", "--dont_plot", action="store_true", required=False,
                        help="don't produce plots")
    parser.add_argument("-g", "--guassian", action="store_true", required=False,
                        help="put the signal in a gaussian")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synth(args)
