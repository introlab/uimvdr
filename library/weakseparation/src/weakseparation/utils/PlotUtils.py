import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_spectrogram_from_spectrogram(spectrogram, title="Spectrogram", show = False):
    # make plot
    fig, ax = plt.subplots()
    fig.suptitle(title)
    # show image
    shw = ax.imshow(20*torch.abs(spectrogram).log10()[:,:].cpu().numpy(), origin='lower')
    
    # make bar
    plt.colorbar(shw)

    if show:
        plt.show()
    else:
        fig.canvas.draw()
    
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def plot_spectrogram_from_waveform(waveform, sample_rate, title="Spectrogram", show=False, xlim=None):
    waveform = waveform.cpu().numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        plot = axes[c].specgram(waveform[c], Fs=sample_rate)
        plt.colorbar(plot[-1])
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

    if show:
        plt.show()
    else:
        figure.canvas.draw()
    
    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    return data


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()