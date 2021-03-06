#!/usr/bin/env python3
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import inspect
import sys
import base_signals
import wave
from scipy.io import wavfile

PI = np.pi
SAMPLERATE = base_signals.SAMPLERATE


def signal_plot(full_axe, partial_axe, signal, fundamental):
    full_axe.set_title("Full Signal")
    full_axe.plot(signal)
    full_axe.set_ylim(-1.1, 1.1)

    partial_axe.set_title("Signal 4 cycles")
    l = list(signal).index(max(signal))
    partial_axe.plot(signal)

    partial_axe.set_xlim(l, l+SAMPLERATE/fundamental*4)
    partial_axe.set_ylim(-1.1, 1.1)
    plt.draw()


def normalize(signal, amp=1.0):
    factor = max(signal)
    return amp*signal / factor


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


def read_file(filename):

    try:
        fs, data = wavfile.read(filename)
    except Exception as e:
        logging.error("Unable to read file {}.\nMay be a malformed wavefile?".format(filename))
        logging.error("Wavfile.read returned: {}".format(e))
        exit(-1)
    if type(data[0]) == type(data):
        logging.info("Found more than one channel, keeping only the first")
        d = data.T[0]
    else:
        d = data
    SAMPLERATE=fs
    return normalize(d)


def play(rawdata):
    data = rawdata.astype(np.float32).tostring()

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLERATE,
                    output=True)
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

def fft(signal, axe=None):
    fft_axe, ceps_axe = axe

    t = np.arange(0,1,1.0/len(signal))
    sp = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.shape[-1], d=1.0/SAMPLERATE)
    spectrum = np.abs(sp)

    peak = max(spectrum)
    fundamental = freq[list(spectrum).index(peak)]
    logging.info("Fundamental is {}".format(fundamental))

    spectrum = normalize(spectrum)

    if fft_axe:
        fft_axe.set_title("FFT")
        fft_axe.set_xlabel("Hz")
        fft_axe.plot(freq, spectrum)
        plt.draw()


    cepstrum = np.fft.irfft(np.log(abs(sp)))

    t = np.arange(0,1,1.0/len(cepstrum))

    if ceps_axe:
        ceps_axe.set_title("cepstrum")
        ceps_axe.set_xlabel("t")
        ceps_axe.plot(t, cepstrum)
        ceps_axe.set_xlim(-0.1, 1.1)
        plt.draw()

    return fundamental


def gen_signal(shape, freq, options):
    signal = options.signalfunct(frequency=freq)

    if options.guassian:
        signal = envelope(signal)

    return normalize(signal)


def scale(options):
    """Produce a PERFECT TEMPERMENT scale
    divide the ocatve into equal parts, which is the 12th roots of 2
    """

    factors = [2**(i/12.0) for i in range(13)]
    if options.major:
        # Major scale is WWHWWWH where W=whole and H=half
        # This can be acomplished by removing steps from the chromatic
        indecies = (0, 2, 4, 5, 7, 9, 11, 12)
        factors = [factors[i] for i in indecies]
    if options.minor:
        indecies = (0, 2, 3, 5, 7, 8, 10, 12)
        factors = [factors[i] for i in indecies]

    freqs = [options.frequency*f for f in factors]
    amps = [1 for f in factors]  # For Aweighting, need to figure this out better

    signals = [options.signalfunct(frequency=f) for f in freqs]
    if options.guassian:
        signals = [a*normalize(envelope(s)) for a, s in zip(amps, signals)]
    else:
        signals = [a*normalize(s) for a, s in zip(amps, signals)]

    singal = np.concatenate(signals)
    logging.info("Playing scale!")
    return singal


def synth(options):
    """ Generate sound from sin waves """
    logging.debug("Called with {}".format(options))

    plot = not options.dont_plot

    if plot:
        fig, plots = plt.subplots(4)
        full_signal_axe, partial_signal_axe, fft_axe, ceps_axe = plots
    else:
        full_signal_axe = partial_signal_axe = fft_axe = ceps_axe = None

    if options.input_file:
        signal = read_file(options.input_file)
        fundamental = fft(signal, (fft_axe, ceps_axe))
    elif options.major or options.chromatic or options.minor:
        signal = scale(options)
        logging.info("Not plotting scale")
        plot = False
    else:
        signal = gen_signal(options.signal, options.frequency, options)
        fundamental = fft(signal, (fft_axe, ceps_axe))

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

    # This is a dirty hack to just read what functions are in
    # "base_signals" and use those as options. so no changes need to
    # be made here if more signals are added
    function_list = inspect.getmembers(sys.modules["base_signals"], inspect.isfunction)
    signal_choices = {name: f for name, f in function_list}

    parser = argparse.ArgumentParser(description="Generate sound from sin waves")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-f", "--frequency", required=False, default=440.0, type=float,
                        help="frequency to generate")
    parser.add_argument("-dp", "--dont_plot", action="store_true", required=False,
                        help="don't produce plots")
    parser.add_argument("-g", "--guassian", action="store_true", required=False,
                        help="put the signal in a gaussian")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")

    signal_group = parser.add_mutually_exclusive_group()
    signal_group.add_argument("-i", "--input_file", type=str, required=False,
                        help="input file to anaylize")
    signal_group.add_argument("-s", "--signal", required=False, choices=signal_choices.keys(),
                        help="which signal to generate", default="harmonics")


    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("-c", "--chromatic", action="store_true", required=False,
                             help="play a chromatic scale")
    scale_group.add_argument("-m", "--major", action="store_true", required=False,
                             help="play a major scale")
    scale_group.add_argument("-r", "--minor", action="store_true", required=False,
                             help="play a minor scale")

    args = parser.parse_args()

    args.signalfunct = signal_choices[args.signal]

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synth(args)
