#!/usr/bin/env python3
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import inspect
import sys
import base_signals


PI = np.pi
SAMPLERATE = base_signals.SAMPLERATE


def signal_plot(full_axe, partial_axe, signal, fundamental):
    full_axe.set_title("Full Signal")
    full_axe.plot(signal)
    full_axe.set_ylim(-1.1, 1.1)

    partial_axe.set_title("Signal 2 cycles")
    partial_axe.plot(signal[0:int(fundamental)])
    partial_axe.set_ylim(-1.1, 1.1)
    plt.draw()


def normalize(signal):
    factor = max(signal)
    return signal / factor


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


def gen_signal(shape, freq, options):
    signal = options.signalfunct(frequency=freq)

    if options.guassian:
        signal = envelope(signal)

    return normalize(signal)


def synth(options):
    """ Generate sound from sin waves """
    logging.debug("Called with {}".format(options))

    plot = not options.dont_plot

    if options.signal == all_sequence:
        signal = all_sequence(options.frequency)
    else:
        signal = gen_signal(options.signal, options.frequency, options)

    if plot:
        fig, plots = plt.subplots(3)
        full_signal_axe, partial_signal_axe, fft_axe = plots
    else:
        full_signal_axe = partial_signal_axe = fft_axe = None

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

    # This is a dirty hack to just read what functions are in
    # "base_signals" and use those as options. so no changes need to
    # be made here if more signals are added
    function_list = inspect.getmembers(sys.modules["base_signals"], inspect.isfunction)
    signal_choices = {name: f for name, f in function_list}

    parser = argparse.ArgumentParser(description="Generate sound from sin waves")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-s", "--signal", required=True, choices=signal_choices.keys(),
                        help="which signal to generate", default="harmonics")
    parser.add_argument("-f", "--frequency", required=False, default=440.0, type=float,
                        help="frequency to generate")
    parser.add_argument("-dp", "--dont_plot", action="store_true", required=False,
                        help="don't produce plots")
    parser.add_argument("-g", "--guassian", action="store_true", required=False,
                        help="put the signal in a gaussian")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    args = parser.parse_args()

    args.signalfunct = signal_choices[args.signal]

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synth(args)
