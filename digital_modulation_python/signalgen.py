import numpy as np

def sine_wave(f, overSampRate, phase, nCyl):
    """
    Generate a sine wave signal with the following parameters
    Parameters:
        f : frequency of the sine wave in Hertz
        overSampRate : oversampling rate (integer)
        phase : desired phase shift in radians
        nCyl : number of cycles of sine wave to generate
    """
    fs = overSampRate * f
    t = np.arange(0, nCyl*1/f - 1/fs, 1/fs)
    g = np.sin(2*np.pi*f*t + phase)
    return (t, g)

def square_wave(f, overSampRate, nCyl):
    fs  = overSampRate * f
    t = np.arange(0, nCyl*1/f - 1/fs, 1/fs)
    g = np.sign(np.sin(2*np.pi*f*t))
    return (t, g)

def rect_pulse(A, fs, T):
    t = np.arange(-0.5, 0.5, 1/fs)
    rect = (t > -T/2) * (t < T/2) + 0.5 * (t==T/2) + 0.5*(t==-T/2)
    g = A * rect
    return (t, g)

def gaussian_pulse(fs, sigma):
    t = np.arange(-0.5, 0.5, 1/fs)
    g = 1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-t**2/(2*sigma**2)))
    return (t, g)