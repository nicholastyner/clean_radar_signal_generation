import numpy as np


def place_pulses(
    pulse,
    num_samples,
    pri_samples,
    pulse_time_delay_samples
):
    """Repeat identical pulses until the waveform buffer is full."""

    pulse = np.asarray(pulse, dtype=np.complex64)

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if pri_samples <= 0:
        raise ValueError("pri_samples must be positive.")

    if pulse_time_delay_samples < 0:
        raise ValueError("Pulse delay cannot be negative.")

    pulse_width_samples = len(pulse)

    if pulse_width_samples == 0:
        raise ValueError("Pulse cannot be empty.")

    if pri_samples < pulse_width_samples:
        raise ValueError("PRI must be at least the pulse width.")

    waveform = np.zeros(num_samples, dtype=np.complex64)

    start = pulse_time_delay_samples

    while start < num_samples:

        end = min(start + pulse_width_samples, num_samples)

        waveform[start:end] = pulse[:end - start]

        start += pri_samples

    return waveform

def distribute_chip_samples(pulse_width_samples, code_length):
    """Distribute a pulse's samples as evenly as possible among chips."""

    if pulse_width_samples < code_length:
        raise ValueError(
            "Pulse width must contain at least one sample per chip."
        )

    base_length = pulse_width_samples // code_length
    remainder = pulse_width_samples % code_length

    chip_lengths = np.full(code_length, base_length, dtype=int)

    chip_lengths[:remainder] += 1

    return chip_lengths