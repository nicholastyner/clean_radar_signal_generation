import numpy as np
from waveform_utils import place_pulses, distribute_chip_samples


def generate_waveform(
    num_samples,
    pulse_width_samples,
    pri_samples,
    pulse_time_delay_samples,
    frank_order=4
):
    if frank_order < 1:
        raise ValueError("Frank order must be positive.")

    code_length = frank_order ** 2

    # Generate Frank phase matrix.
    frank_code = np.zeros(
        (frank_order, frank_order),
        dtype=np.float64
    )

    for p in range(frank_order):
        for q in range(frank_order):
            frank_code[p, q] = (
                2 * np.pi * p * q / frank_order
            )

    # Flatten the matrix into a one-dimensional code.
    phase_sequence = frank_code.flatten()

    frank_sequence = np.exp(1j * phase_sequence)

    chip_lengths = distribute_chip_samples(
        pulse_width_samples,
        code_length
    )

    pulse = np.repeat(frank_sequence, chip_lengths)

    waveform = place_pulses(
        pulse,
        num_samples,
        pri_samples,
        pulse_time_delay_samples
    )

    return waveform.astype(np.complex64)