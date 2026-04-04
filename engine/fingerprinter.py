import numpy as np

def compute_dft(signal):
    """ For computing the Discrete Transorm of 1D signal"""
    N=len(signal)
    X=np.zeros(N,dtype=complex)
    for k in range(N):
        for n in range(N):
            angle=-2j*np.pi*k*n/N
            X[k]+=signal[n]*np.exp(angle)
    return X


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    sr=100
    t=np.linspace(0,1,sr,endpoint=False)
    wave_10hz=np.sin(2*np.pi*10*t)
    wave_20hz=np.sin(2*np.pi*20*t)
    signal=wave_10hz+wave_20hz

    print("Computing DFT...")
    X=compute_dft(signal)
    magnitudes=np.abs(X)

    plt.plot(magnitudes[:sr//2])
    plt.title("Frequency Spectrum (DFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()
