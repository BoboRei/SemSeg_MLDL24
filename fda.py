import torch
import numpy as np

def extract_ampl_phase(fft_im):
    fft_real = fft_im.real
    fft_imag = fft_im.imag
    fft_amp = torch.sqrt(fft_real**2 + fft_imag**2)
    fft_pha = torch.atan2(fft_imag, fft_real)
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, beta=0.1):
    _, _, h, w = amp_src.size()
    b = int(np.floor(min(h, w) * beta))
    amp_src[:, :, :b, :b] = amp_trg[:, :, :b, :b]
    amp_src[:, :, :b, -b:] = amp_trg[:, :, :b, -b:]
    amp_src[:, :, -b:, :b] = amp_trg[:, :, -b:, :b]
    amp_src[:, :, -b:, -b:] = amp_trg[:, :, -b:, -b:]
    return amp_src

def FDA_source_to_target(src_img, trg_img, beta=0.1):
    if trg_img.size(0) < src_img.size(0):
        trg_img = trg_img[:src_img.size(0)]
    
    # Get fft of both source and target
    fft_src = torch.fft.rfft2(src_img, dim=(-2, -1))
    fft_trg = torch.fft.rfft2(trg_img, dim=(-2, -1))

    # Extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)

    # Troncare amp_trg se necessario
    if amp_trg.size(0) != amp_src.size(0):
        amp_trg = amp_trg[:amp_src.size(0)]

    # Replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src, amp_trg, beta=beta)

    # Recompose fft of source
    fft_src_ = torch.polar(amp_src_, pha_src)

    # Get the recomposed image: source content, target style
    src_in_trg = torch.fft.irfft2(fft_src_, s=src_img.shape[-2:])
    return src_in_trg