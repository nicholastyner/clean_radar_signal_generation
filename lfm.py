import numpy as np
from waveform_utils import place_pulses


def generate_waveform(
    num_samples,
    pulse_width_samples,
    pri_samples,
    pulse_time_delay_samples,
    sample_rate=3.125e6,
    bandwidth=0.3770e6
):
    if pulse_width_samples <= 0:
        raise ValueError("Pulse width must be positive.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if bandwidth <= 0 or bandwidth >= sample_rate:
        raise ValueError(
            "Bandwidth must be positive and less than the sample rate."
        )

    pulse_duration = pulse_width_samples / sample_rate

    t = np.arange(pulse_width_samples) / sample_rate

    # Linear frequency sweep from -bandwidth/2 to +bandwidth/2.
    start_frequency = -bandwidth / 2

    chirp_rate = bandwidth / pulse_duration

    phase = (
        2 * np.pi * start_frequency * t
        + np.pi * chirp_rate * t**2
    )

    pulse = np.exp(1j * phase).astype(np.complex64)

    waveform = place_pulses(
        pulse,
        num_samples,
        pri_samples,
        pulse_time_delay_samples
    )

    return waveform