import os
import random
import numpy as np

# Import waveform generators from individual files.
from coherent import generate_waveform as generate_coherent_waveform
from barker import generate_waveform as generate_barker_waveform
from polyphase_barker import generate_waveform as generate_polyphase_barker_waveform
from frank import generate_waveform as generate_frank_waveform
from lfm import generate_waveform as generate_lfm_waveform


# ============================================================
# CONFIGURATION
# ============================================================

CAPTURES = 100
SAMPLES_PER_WAVEFORM = 512
SAMPLE_RATE = 3.125e6

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)

RANDOM_SEED = None

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


# ============================================================
# RANDOM PARAMETER RANGES
# ============================================================

PULSE_WIDTH_MIN = 10e-6
PULSE_WIDTH_MAX = 16e-6

PRI_MIN = 17e-6
PRI_MAX = 23e-6

PULSE_TIME_DELAY_MIN = 1e-6
PULSE_TIME_DELAY_MAX = 10e-6

LFM_BANDWIDTH_MIN = 0.15e6
LFM_BANDWIDTH_MAX = 0.6e6

BARKER_CODE_LENGTHS = [2, 3, 4, 5, 7, 11, 13]

POLYPHASE_BARKER_CODE_LENGTHS = [2, 3, 4, 5, 7, 11, 13]

FRANK_ORDERS = [2, 3, 4]


SIGNAL_TYPES = [
    "coherent",
    "barker",
    "polyphase_barker",
    "frank",
    "lfm"
]


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

def create_directories():

    for signal_type in SIGNAL_TYPES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                signal_type,
                "data",
                "raw"
            ),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                signal_type,
                "data",
                "normalized"
            ),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                signal_type,
                "labels"
            ),
            exist_ok=True
        )


# ============================================================
# RANDOM PARAMETER GENERATION
# ============================================================

def generate_values():

    # Generate pulse width.
    pulse_width = random.uniform(
        PULSE_WIDTH_MIN,
        PULSE_WIDTH_MAX
    )

    pulse_width_samples = round(
        pulse_width * SAMPLE_RATE
    )

    # Generate PRI while ensuring pulses do not overlap.
    minimum_pri = max(
        PRI_MIN,
        pulse_width
    )

    if minimum_pri > PRI_MAX:
        raise ValueError(
            "Pulse width exceeds maximum allowed PRI."
        )

    pri = random.uniform(
        minimum_pri,
        PRI_MAX
    )

    pri_samples = round(
        pri * SAMPLE_RATE
    )

    # Enforce non-overlap after sample conversion.
    pri_samples = max(
        pri_samples,
        pulse_width_samples
    )

    # Generate initial pulse delay.
    pulse_time_delay = random.uniform(
        PULSE_TIME_DELAY_MIN,
        PULSE_TIME_DELAY_MAX
    )

    pulse_time_delay_samples = round(
        pulse_time_delay * SAMPLE_RATE
    )

    return {
        "pulse_width_seconds": pulse_width_samples / SAMPLE_RATE,
        "pulse_width_samples": pulse_width_samples,

        "pri_seconds": pri_samples / SAMPLE_RATE,
        "pri_samples": pri_samples,

        "pulse_time_delay_seconds": (
            pulse_time_delay_samples / SAMPLE_RATE
        ),
        "pulse_time_delay_samples": pulse_time_delay_samples
    }


# ============================================================
# PULSE COUNT
# ============================================================

def calculate_actual_num_pulses(
    num_samples,
    pri_samples,
    pulse_time_delay_samples
):
    """
    Count pulses whose starting positions fall within the waveform.

    The final pulse is counted even if it is truncated by the
    end of the waveform buffer.
    """

    if pulse_time_delay_samples >= num_samples:
        return 0

    return (
        (num_samples - 1 - pulse_time_delay_samples)
        // pri_samples
    ) + 1


# ============================================================
# WAVEFORM GENERATORS
# ============================================================

def generate_coherent(parameters):

    waveform = generate_coherent_waveform(
        num_samples=SAMPLES_PER_WAVEFORM,
        pulse_width_samples=parameters["pulse_width_samples"],
        pri_samples=parameters["pri_samples"],
        pulse_time_delay_samples=parameters[
            "pulse_time_delay_samples"
        ]
    )

    return waveform, {}


def generate_barker(parameters):

    code_length = random.choice(
        BARKER_CODE_LENGTHS
    )

    waveform = generate_barker_waveform(
        num_samples=SAMPLES_PER_WAVEFORM,
        pulse_width_samples=parameters["pulse_width_samples"],
        pri_samples=parameters["pri_samples"],
        pulse_time_delay_samples=parameters[
            "pulse_time_delay_samples"
        ],
        barker_code_length=code_length
    )

    specific_labels = {
        "barker_code_length": code_length
    }

    return waveform, specific_labels


def generate_polyphase_barker(parameters):

    code_length = random.choice(
        POLYPHASE_BARKER_CODE_LENGTHS
    )

    waveform = generate_polyphase_barker_waveform(
        num_samples=SAMPLES_PER_WAVEFORM,
        pulse_width_samples=parameters["pulse_width_samples"],
        pri_samples=parameters["pri_samples"],
        pulse_time_delay_samples=parameters[
            "pulse_time_delay_samples"
        ],
        code_length=code_length
    )

    specific_labels = {
        "polyphase_barker_code_length": code_length,
        "phase_alphabet": 4
    }

    return waveform, specific_labels


def generate_frank(parameters):

    order = random.choice(
        FRANK_ORDERS
    )

    waveform = generate_frank_waveform(
        num_samples=SAMPLES_PER_WAVEFORM,
        pulse_width_samples=parameters["pulse_width_samples"],
        pri_samples=parameters["pri_samples"],
        pulse_time_delay_samples=parameters[
            "pulse_time_delay_samples"
        ],
        frank_order=order
    )

    specific_labels = {
        "frank_order": order,
        "frank_code_length": order ** 2
    }

    return waveform, specific_labels


def generate_lfm(parameters):

    bandwidth = random.uniform(
        LFM_BANDWIDTH_MIN,
        LFM_BANDWIDTH_MAX
    )

    waveform = generate_lfm_waveform(
        num_samples=SAMPLES_PER_WAVEFORM,
        pulse_width_samples=parameters["pulse_width_samples"],
        pri_samples=parameters["pri_samples"],
        pulse_time_delay_samples=parameters[
            "pulse_time_delay_samples"
        ],
        sample_rate=SAMPLE_RATE,
        bandwidth=bandwidth
    )

    pulse_duration = (
        parameters["pulse_width_samples"] / SAMPLE_RATE
    )

    chirp_rate = bandwidth / pulse_duration

    specific_labels = {
        "lfm_bandwidth_hz": bandwidth,
        "lfm_chirp_rate_hz_per_second": chirp_rate
    }

    return waveform, specific_labels


# ============================================================
# DATA STORAGE
# ============================================================

def save_waveform(
    signal_type,
    capture_number,
    waveform,
    parameters,
    actual_num_pulses,
    specific_labels
):

    waveform = np.asarray(
        waveform,
        dtype=np.complex64
    )

    if waveform.shape != (SAMPLES_PER_WAVEFORM,):
        raise ValueError(
            f"Invalid waveform shape: {waveform.shape}"
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            f"{signal_type} waveform contains NaN or Inf."
        )

    peak = np.max(np.abs(waveform))

    if peak == 0:
        raise ValueError(
            f"{signal_type} waveform is all zeros."
        )

    normalized = waveform / peak

    filename = f"{capture_number:03d}.txt"

    raw_path = os.path.join(
        OUTPUT_DIR,
        signal_type,
        "data",
        "raw",
        filename
    )

    normalized_path = os.path.join(
        OUTPUT_DIR,
        signal_type,
        "data",
        "normalized",
        filename
    )

    label_path = os.path.join(
        OUTPUT_DIR,
        signal_type,
        "labels",
        filename
    )

    # Save two columns: I and Q.
    raw_iq = np.column_stack((
        waveform.real,
        waveform.imag
    ))

    normalized_iq = np.column_stack((
        normalized.real,
        normalized.imag
    ))

    np.savetxt(
        raw_path,
        raw_iq,
        fmt="%.10e",
        header="I Q"
    )

    np.savetxt(
        normalized_path,
        normalized_iq,
        fmt="%.10e",
        header="I Q"
    )

    all_labels = {
        "signal_type": signal_type,
        "sample_rate_hz": int(SAMPLE_RATE),
        "num_samples": SAMPLES_PER_WAVEFORM,

        **parameters,

        "actual_num_pulses": actual_num_pulses,

        **specific_labels
    }

    with open(label_path, "w") as file:

        for key, value in all_labels.items():
            file.write(
                f"{key}: {value}\n"
            )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def main():

    print("=" * 65)
    print("RADAR WAVEFORM DATASET GENERATION")
    print("=" * 65)

    create_directories()

    generators = {
        "coherent": generate_coherent,
        "barker": generate_barker,
        "polyphase_barker": generate_polyphase_barker,
        "frank": generate_frank,
        "lfm": generate_lfm
    }

    total_waveforms = 0

    for signal_type in SIGNAL_TYPES:

        print(f"\nGenerating {signal_type}...")

        generator = generators[signal_type]

        for capture_number in range(1, CAPTURES + 1):

            # Generate independent physical parameters.
            parameters = generate_values()

            # Generate waveform using its individual module.
            waveform, specific_labels = generator(parameters)

            # Count pulses from the actual placement parameters.
            actual_num_pulses = calculate_actual_num_pulses(
                SAMPLES_PER_WAVEFORM,
                parameters["pri_samples"],
                parameters["pulse_time_delay_samples"]
            )

            # Save waveform and labels.
            save_waveform(
                signal_type=signal_type,
                capture_number=capture_number,
                waveform=waveform,
                parameters=parameters,
                actual_num_pulses=actual_num_pulses,
                specific_labels=specific_labels
            )

            total_waveforms += 1

            print(
                f"  {capture_number:03d}/{CAPTURES} "
                f"| pulses: {actual_num_pulses}"
            )

    print("\n" + "=" * 65)
    print("DATASET GENERATION COMPLETE")
    print("=" * 65)

    print(f"Signal types: {len(SIGNAL_TYPES)}")
    print(f"Captures per type: {CAPTURES}")
    print(f"Total waveforms: {total_waveforms}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()