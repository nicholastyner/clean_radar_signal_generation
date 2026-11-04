import numpy as np
from waveform_utils import place_pulses, distribute_chip_samples


BARKER_CODES = {
    2:  [1, -1],
    3:  [1, 1, -1],
    4:  [1, 1, -1, 1],
    5:  [1, 1, 1, -1, 1],
    7:  [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
}


def generate_waveform(
    num_samples,
    pulse_width_samples,
    pri_samples,
    pulse_time_delay_samples,
    code_length
):
    if code_length not in BARKER_CODES:
        raise ValueError("Unsupported code length.")

    barker_code = np.array(BARKER_CODES[code_length])

    # Generate a four-phase Barker-derived sequence.
    chip_indices = np.arange(code_length)

    phase_indices = (
        chip_indices + 2 * (barker_code < 0)
    ) % 4

    phase_values = phase_indices * (np.pi / 2)

    polyphase_code = np.exp(1j * phase_values)

    chip_lengths = distribute_chip_samples(
        pulse_width_samples,
        code_length
    )

    pulse = np.repeat(polyphase_code, chip_lengths)

    waveform = place_pulses(
        pulse,
        num_samples,
        pri_samples,
        pulse_time_delay_samples
    )

    return waveform.astype(np.complex64)