import numpy as np
from waveform_utils import place_pulses


def generate_waveform(
    num_samples,
    pulse_width_samples,
    pri_samples,
    pulse_time_delay_samples
):
    if pulse_width_samples <= 0:
        raise ValueError("Pulse width must be positive.")

    pulse = np.ones(pulse_width_samples, dtype=np.complex64)

    waveform = place_pulses(
        pulse,
        num_samples,
        pri_samples,
        pulse_time_delay_samples
    )

    return waveform