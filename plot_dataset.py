
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

SIGNAL_TYPES = [
    "coherent",
    "barker",
    "polyphase_barker",
    "frank",
    "lfm"
]

CAPTURES = 100

# Initial display settings
CAPTURE_NUMBER = 1
SIGNAL_TYPE_INDEX = 0

# Choose "raw" or "normalized"
DATA_VARIANT = "raw"

SAMPLES_PER_WAVEFORM = 512


# ============================================================
# LOAD WAVEFORM
# ============================================================

def load_waveform(signal_type, capture_number, data_variant):

    file_path = (
        DATASET_DIR
        / signal_type
        / "data"
        / data_variant
        / f"{capture_number:03d}.txt"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    iq_data = np.loadtxt(file_path)

    if iq_data.shape != (SAMPLES_PER_WAVEFORM, 2):
        raise ValueError(
            f"Expected ({SAMPLES_PER_WAVEFORM}, 2) IQ data, "
            f"got {iq_data.shape} from {file_path}"
        )

    return iq_data[:, 0] + 1j * iq_data[:, 1]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

sample_indices = np.arange(SAMPLES_PER_WAVEFORM)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 8),
    sharex=True
)

fig.subplots_adjust(
    left=0.09,
    right=0.97,
    bottom=0.10,
    top=0.82,
    hspace=0.38
)

ax_i, ax_q, ax_mag = axes

# Create plot lines once so they can be updated without clearing axes
line_i, = ax_i.plot([], [], linewidth=1.2)
line_q, = ax_q.plot([], [], linewidth=1.2)
line_mag, = ax_mag.plot([], [], linewidth=1.2)

ax_i.set_ylabel("I Amplitude")
ax_q.set_ylabel("Q Amplitude")
ax_mag.set_ylabel("Magnitude")

ax_mag.set_xlabel("Sample Index")

for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, SAMPLES_PER_WAVEFORM - 1)
    ax.set_xticks(np.arange(0, SAMPLES_PER_WAVEFORM, 25))


# ============================================================
# FIGURE TEXT
# ============================================================

title_text = fig.suptitle(
    "",
    fontsize=15,
    fontweight="bold",
    y=0.96
)

navigation_text = fig.text(
    0.5,
    0.035,
    "← Previous capture    |    → Next capture"
    "    |    ↓ Previous waveform type"
    "    |    ↑ Next waveform type",
    ha="center",
    fontsize=9,
    color="gray"
)


# ============================================================
# UPDATE PLOTS
# ============================================================

def update_plot():

    signal_type = SIGNAL_TYPES[SIGNAL_TYPE_INDEX]

    received = load_waveform(
        signal_type,
        CAPTURE_NUMBER,
        DATA_VARIANT
    )

    i_data = received.real
    q_data = received.imag
    magnitude = np.abs(received)

    # Update existing plot lines
    line_i.set_data(sample_indices, i_data)
    line_q.set_data(sample_indices, q_data)
    line_mag.set_data(sample_indices, magnitude)

    # Recalculate axis limits
    for ax, values in zip(
        axes,
        [i_data, q_data, magnitude]
    ):
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax.set_xlim(0, SAMPLES_PER_WAVEFORM - 1)

    # Update title
    title_text.set_text(
        f"{signal_type.replace('_', ' ').title()} | "
        f"Capture {CAPTURE_NUMBER:03d}/{CAPTURES} | "
        f"Received: {DATA_VARIANT}"
    )

    # Display statistics
    peak_magnitude = np.max(magnitude)
    rms_magnitude = np.sqrt(np.mean(magnitude ** 2))

    print("\n" + "=" * 55)
    print(f"Signal type: {signal_type}")
    print(f"Capture: {CAPTURE_NUMBER:03d}")
    print(f"Received data: {DATA_VARIANT}")
    print(f"Peak magnitude: {peak_magnitude:.6f}")
    print(f"RMS magnitude:  {rms_magnitude:.6f}")
    print("=" * 55)

    fig.canvas.draw_idle()


# ============================================================
# KEYBOARD CONTROLS
# ============================================================

def on_key(event):

    global CAPTURE_NUMBER
    global SIGNAL_TYPE_INDEX

    if event.key == "right":

        CAPTURE_NUMBER += 1

        if CAPTURE_NUMBER > CAPTURES:
            CAPTURE_NUMBER = 1

    elif event.key == "left":

        CAPTURE_NUMBER -= 1

        if CAPTURE_NUMBER < 1:
            CAPTURE_NUMBER = CAPTURES

    elif event.key == "up":

        SIGNAL_TYPE_INDEX += 1

        if SIGNAL_TYPE_INDEX >= len(SIGNAL_TYPES):
            SIGNAL_TYPE_INDEX = 0

    elif event.key == "down":

        SIGNAL_TYPE_INDEX -= 1

        if SIGNAL_TYPE_INDEX < 0:
            SIGNAL_TYPE_INDEX = len(SIGNAL_TYPES) - 1

    else:
        return

    update_plot()


# ============================================================
# INITIALIZE
# ============================================================

fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)

print("DATASET WAVEFORM VIEWER")
print("-----------------------")
print("Right arrow: Next capture")
print("Left arrow: Previous capture")
print("Up arrow: Next waveform type")
print("Down arrow: Previous waveform type")
print("Click the plot window before using arrow keys.")
print("Close the plot window to exit.")

update_plot()

plt.show()