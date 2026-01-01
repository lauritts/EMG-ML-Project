import numpy as np
from model import (
    mean_absolute_value as time_mav,
    waveform_length as time_wl,
    zero_crossings as time_zc,
    slope_sign_changes as time_ssc,
    root_mean_square as time_rms,
    spectral_moments, spectral_flux, spectral_sparsity,
    spectral_irregularity, spectral_correlation
)


def make_signal(channels=4, length=512, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, length)
    sig = np.zeros((channels, length), dtype=float)
    for c in range(channels):
        freq = 5 + c * 3
        sig[c] = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.randn(length)
    return sig


def run_all():
    x = make_signal(4, 512)

    # time features
    mav = time_mav(x)
    wl = time_wl(x)
    zc = time_zc(x)
    ssc = time_ssc(x)
    rms = time_rms(x)

    assert mav.shape == (4,)
    assert wl.shape == (4,)
    assert zc.shape == (4,)
    assert ssc.shape == (4,)
    assert rms.shape == (4,)

    # spectral features
    mom = spectral_moments(x, fs=512)
    sf = spectral_flux(x, fs=512)
    sp = spectral_sparsity(x, fs=512)
    si = spectral_irregularity(x, fs=512)
    sc = spectral_correlation(x, fs=512)

    assert mom.shape[0] == 4
    assert sf.shape == (4,)
    assert sp.shape == (4,)
    assert si.shape == (4,)
    assert sc.shape == (4,)

    print('All extractor smoke tests passed')


if __name__ == '__main__':
    run_all()
