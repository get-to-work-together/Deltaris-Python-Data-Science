import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# DFFT
def f(t, harmonics):
    v = 0
    for i, harmonic in enumerate(harmonics):
        v += harmonic * np.sin(i * 2.0 * np.pi * t)
    return v

# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 30.0

t = np.linspace(0.0, N*T, N)

harmonics = (0, 1.0, 0.5, 0.25, 0.125, 0, 0, 0, 0)
y = f(t, harmonics)

plt.plot(t, y)
plt.show()
# %% FFT

yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)

ax[0].plot(t, y)
ax[0].set_yticks([0])
ax[0].grid(True, which='major', axis='y')
ax[0].set_title('Signal')
ax[1].plot(xf, 2.0/N * np.abs(yf[:N//2]), color='#dd4444')

ax[1].set_title('Spectrum')
ax[1].grid(True)
ax[1].set_xlim(xmin=0)
ax[1].set_ylim(ymin=0)
ax[1].set_xticks(np.arange(*ax[1].get_xlim(), step=1))
ax[1].set_yticks(np.arange(*ax[1].get_ylim(), step=0.2))

plt.show()
