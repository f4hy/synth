#!/usr/bin/env python3
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import os
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
import itertools
import pyaudio

PI=np.pi

SAMPLERATE = 44100


def signal_plot(axe, signal, fundamental):
    axe.set_title("Signal")
    axe.plot(signal[0:500])
    axe.set_ylim(-1.1, 1.1)
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
    for n in range(1,number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)

    if norm:
        return normalize(signal)
    else:
        return signal

def even_harmonics(frequency=440.0, number=50, norm=True):
    logging.info("Creating signal and the first {} harmonics of {}".format(number, frequency))
    amp = 1.0
    decay = 0.6
    signal = sine_wave(frequency=frequency, amplitude=amp)
    nth_frequncy = frequency
    for n in range(1,number):
        nth_amp = amp*np.exp(-1.0*decay*n)
        nth_frequncy += 2*frequency
        signal = signal+sine_wave(frequency=nth_frequncy, amplitude=nth_amp)

    if norm:
        return normalize(signal)
    else:
        return signal



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
    m = list(map(minus_or_one, np.sin(t*frequency)))
    vfun = np.vectorize(minus_or_one)
    s = vfun(np.sin(t*frequency))
    return s

def play(rawdata):
    #print([int(rawdata[x]*127+128) for x in range(SAMPLERATE)])

    # to play audio it takes values between 0 and 128, but streamed in
    # as characters.
    # so data that is [-1.0 -0.5 0.0 0.5 1.0] needs to
    # be [0 32 64 96 128] streamed in as a string so the chr() for
    # those values
    data = ''.join([chr(int(rawdata[x]*127+128)) for x in range(SAMPLERATE)])

    p = pyaudio.PyAudio()

    stream = p.open(format =
                    p.get_format_from_width(1),
                    channels = 1,
                    rate = SAMPLERATE,
                    output = True)
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

def fft(signal, axe=None):
    t = np.arange(256)
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
        signal = even_harmonics(number=50, frequency=440.0)
    if options.signal == "square":
        signal = square_wave()
    if options.signal == "sine":
        signal = sine_wave()

    if plot:
        fig, plots = plt.subplots(2)
        signal_axe, fft_axe = plots
    else:
        signal_axe = fft_axe = None


    fundamental = fft(signal, fft_axe)
    play(signal)

    signal_plot(signal_axe, signal, fundamental)

    if plot:
        if(options.output_stub):
            f.set_size_inches(18.5, 18.5)
            ofilename = output_stub+".png"
            logging.info("Saving plot to {}".format(ofilename))
            plt.savefig(ofilename, dpi=400)
            return
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound from sin waves")
    signal_choices = ["sine", "square", "harmonics", "even-harmonics"]
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-s", "--signal", required=True, choices=signal_choices,
                        help="which signal to generate", default="harmonics")
    parser.add_argument("-dp", "--dont_plot", required=False, default=False,
                        help="don't produce plots")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synth(args)
