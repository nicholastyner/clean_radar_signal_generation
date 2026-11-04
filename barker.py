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
    barker_code_length
):
    if barker_code_length not in BARKER_CODES:
        raise ValueError("Unsupported Barker code length.")

    barker_code = np.array(
        BARKER_CODES[barker_code_length],
        dtype=np.complex64
    )

    chip_lengths = distribute_chip_samples(
        pulse_width_samples,
        barker_code_length
    )

    pulse = np.repeat(barker_code, chip_lengths)

    waveform = place_pulses(
        pulse,
        num_samples,
        pri_samples,
        pulse_time_delay_samples
    )

    return waveform