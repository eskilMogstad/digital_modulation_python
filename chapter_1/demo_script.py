
import numpy as np
import matplotlib.pyplot as plt

def sine_wave_demo():
    from digital_modulation_python.signalgen import sine_wave

    f = 10
    overSampRate = 30
    phase = 1/3*np.pi
    nCyl = 5
    (t, g) = sine_wave(f, overSampRate, phase, nCyl)

    plt.plot(t, g)
    plt.title("Sine wave f=" + str(f) + " Hz")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

def square_wave_demo():
    from digital_modulation_python.signalgen import square_wave

    f = 10
    overSampRate = 30
    nCyl = 5
    (t, g) = square_wave(f, overSampRate, nCyl)

    plt.plot(t, g)
    plt.title("Square wave f=" + str(f) + " Hz")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

def scipy_square_wave():
    from scipy import signal

    f = 10
    overSampRate = 30
    nCyl = 5

    fs = overSampRate * f
    t = np.arange(start=0, stop=nCyl*1/f, step=1/fs)
    g = signal.square(2 * np.pi * f * t, duty = 0.2)
    plt.plot(t, g)
    plt.show()

def chirp_demo():
    from scipy.signal import chirp

    fs = 500
    t = np.arange(start=0, stop=1, step=1/fs)
    g = chirp(t, f0=1, t1=0.5, f1=20, phi=0, method='linear')
    plt.plot(t,g)
    plt.show()

def compare_convutions():
    from scipy.fftpack import fft, ifft
    from digital_modulation_python.essentials import my_convolve

    x = np.random.normal(size = 7) + 1j * np.random.normal(size = 7)
    h = np.random.normal(size = 3) + 1j * np.random.normal(size = 3)
    L = len(x) + len(h) - 1

    y1 = my_convolve(h, x)
    y2 = ifft(fft(x, L)*(fft(h, L).T))
    y3 = np.convolve(h, x)
    print(f' y1 : {y1} \n  y2 : {y2} \n  y3 : {y3} \n ')

def analytic_signal_demo():
    from digital_modulation_python.essentials import analytic_signal
    
    t = np.arange(start=0, stop=0.5, step=0.001)
    x = np.sin(2*np.pi*10*t)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(t, x)
    ax1.set_title("x[n] - real-valued signal")
    ax1.set_xlabel("n")
    ax1.set_ylabel("x[n]")

    z = analytic_signal(x)

    ax2.plot(t, np.real(z), 'k', label="Real(z[n])")
    ax2.plot(t, np.imag(z), 'r', label="Imag(z[n])")
    ax2.set_title("Components of analytic signal")
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$z_r[n]$ and $z_i[n]$")
    ax2.legend()

    fig.show()
    plt.show()

def extract_envelope_phase():
    from scipy.signal import chirp
    from digital_modulation_python.essentials import analytic_signal
    
    fs = 600
    t = np.arange(start=0, stop=1, step=1/fs)
    a_t = 1.0 + 0.7*np.sin(2.0*np.pi*3.0*t)
    c_t = chirp(t, f0=20, t1=t[-1], f1=80, phi=0, method='linear')
    x = a_t * c_t
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(x)

    z = analytic_signal(x)
    inst_amplitude = abs(z)
    inst_phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
    
    extracted_carrier = np.cos(inst_phase)

    ax1.plot(inst_amplitude, 'r')
    ax1.set_title("Modulated signal and extracted envelope")
    ax1.set_xlabel("n")
    ax1.set_ylabel(r"x(t) and $|z(t)|$")

    ax2.plot(extracted_carrier)
    ax2.set_title("Extracted carrier or TFS")
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$cos[\omega(t)]")

    fig.show()
    plt.show()

def hilbert_phase_demod():
    from scipy.signal import hilbert
    fc = 210
    fm = 10
    alpha = 1
    theta = np.pi/4
    beta = np.pi/5

    receiverKnowsCarrier = False

    fs = 8 * fc
    duration = 0.5
    t = np.arange(start=0, stop=duration, step=1/fs)

    m_t = alpha*np.sin(2*np.pi*fm*t + theta)
    x = np.cos(2*np.pi*fc*t + beta + m_t)

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t, m_t)
    ax1.set_title("Modulating signal")
    ax1.set_xlabel("t")
    ax1.set_ylabel("m(t)")
    
    ax2.plot(t, x)
    ax2.set_title("Modulated signal")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x(t)")
    
    fig1.show()

    mu = 0
    sigma = 0.1
    n = mu + sigma*np.random.normal(size=len(t))
    r = x + n

    z = hilbert(r)
    inst_phase = np.unwrap(np.angle(z))

    if receiverKnowsCarrier:
        offsetTerm = 2*np.pi*fc*t + beta
    else:
        p = np.polyfit(x=t, y=inst_phase, deg=1)
        estimated = np.polyval(p, t)
        offsetTerm = estimated

    demodulated = inst_phase - offsetTerm

    fig2, ax3 = plt.subplots()

    ax3.plot(t, demodulated)
    ax3.set_title("Demodulated signal")
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\hat{m(t)}$")
    
    fig2.show()
    plt.show()


def main():
    #sine_wave_demo()
    #square_wave_demo()
    #scipy_square_wave()
    #chirp_demo()
    #compare_convutions()
    #analytic_signal_demo()
    #extract_envelope_phase()
    hilbert_phase_demod()
    print("a")

if __name__ == "__main__":
    main()