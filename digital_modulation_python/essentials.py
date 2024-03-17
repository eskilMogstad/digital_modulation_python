import numpy as np

def plotWelchPSD(x, fs, fc, ax = None, color = 'b', label = None):
    from scipy.signal import hanning, welch

    nx = max(x.shape)
    na = 16
    w = hanning(nx // na)

    f, Pxx = welch(x, fs, window = w, noverlap=0)
    indices = (f >= fs) & (f < 4*fc)
    Pxx = Pxx[indices]/Pxx[indices][0]
    ax.plot(f[indices] - fc, 10*np.log10(Pxx), color, label=label)

def conv_brute_force(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)

    for i in np.arange(0, N):
        for j in np.arange(0, M):
            y[i + j] = y[i + j] + x[i] * h[j]

    return y

def convMatrix(h, p):
    from scipy.linalg import toeplitz
    
    col = np.hstack((h, np.zeros(p - 1)))
    row = np.hstack((h[0], np.zeros(p - 1)))

    H = toeplitz(col, row)
    
    return H

def my_convolve(h, x):
    H = convMatrix(h, len(x))
    
    y = H @ x.transpose()

    return y

def analytic_signal(x):
    from scipy.fftpack import fft, ifft

    N = len(x)
    X = fft(x, N)
    Z = np.hstack((X[0], 2*X[1:N//2], X[N//2], np.zeros(N//2-1)))
    z = ifft(Z, N)
    return z 
