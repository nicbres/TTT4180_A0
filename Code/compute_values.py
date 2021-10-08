#!./venv/bin/python3
import csv
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig


ntnu_dir_path = pathlib.Path('/home') / 'nbresina' / 'Documents' / 'NTNU'
assignment_path = ntnu_dir_path / 'TechnicalAcoustics' / 'Assignments' / '0'

recordings_path = assignment_path / 'Recordings'
samfundet_folder_path = recordings_path / 'Samfundet'
samfundet_ref_pre_path = samfundet_folder_path / 'ZOOM0004_reference_pre' / 'ZOOM0004_Tr1.WAV'
samfundet_rec_path = samfundet_folder_path / 'ZOOM0007_samfundet' / 'ZOOM0007_Noise_Rec_30min.wav'
samfundet_ref_post_path = samfundet_folder_path / 'ZOOM0008_reference_post' / 'ZOOM0008_Tr1.WAV'

code_path = assignment_path / 'Code'
a_weights_csv = code_path / 'a_weights.csv'
third_octave_csv = code_path / 'third_octave_frequencies.csv'


class Recording:
    pressure_reference = 20*10**(-6)
    pressure_calibrator = 10**(94/20)*pressure_reference

    def __init__(
        self,
        file_path,
        calibration_factor_pre=1.6831914497821373e-09,
        calibration_factor_post=1.8531190247712699e-09,
    ):
        """
        Generates a Recording object from a file using the provided path. It uses
        hardcoded defaults which were computed from the reference recordings.
        """
        logging.info("------------- Reading WAV files -------------")
        self._samplerate, self._recording = wav.read(file_path)

        self._cal_factor_pre = calibration_factor_pre
        self._cal_factor_post = calibration_factor_post

    @classmethod
    def from_recordings(
        cls,
        recording_path: pathlib.Path,
        reference_pre_recording_path: pathlib.Path,
        reference_post_recording_path: pathlib.Path,
    ):
        """
        This constructor can be used to compute the calibration factors from
        the given recordings and instantiate a Recording object with the
        resulting values.
        """
        # Read Data
        logging.info("------------- Reading Reference Files -------------")
        samplerate, ref_pre = wav.read(reference_pre_recording_path)
        logging.info(f"ref_pre samplerate: {samplerate}")
        samplerate,  ref_post = wav.read(reference_post_recording_path)
        logging.info(f"ref_post samplerate: {samplerate}")

        # Calculate RMS in Digital-Time/Sample Domain
        logging.info("--------------- Digital RMS -----------------")
        digital_ref_rms_pre = root_mean_square(
            sampled_data=ref_pre,
        )
        logging.info(f"Digital RMS Pre: {digital_ref_rms_pre:.3e}")

        digital_ref_rms_post = root_mean_square(
            sampled_data=ref_post,
        )
        logging.info(f"Digital RMS Post: {digital_ref_rms_post:.3e}")

        # Calculate Calibration Factors from the Reference Recordings
        logging.info("----------- Calibration Factor --------------")
        calibration_factor_pre = calibration_factor(
            digital_ref_rms=digital_ref_rms_pre,
            physical_ref_rms=cls.pressure_calibrator,
        )
        logging.info(f"Calibration Factor Pre: {calibration_factor_pre:.3e}")

        calibration_factor_post = calibration_factor(
            digital_ref_rms=digital_ref_rms_post,
            physical_ref_rms=cls.pressure_calibrator,
        )
        logging.info(f"Calibration Factor Post: {calibration_factor_post:.3e}")

        return Recording(
            file_path=recording_path,
            calibration_factor_pre=calibration_factor_pre,
            calibration_factor_post=calibration_factor_post,
        )

    @property
    def recording(self):
        return self._recording

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def calibrated_pre(self):
        """
        Returns the pressure p[n] calibrated with the reference pre recording
        """
        return self._recording*self._cal_factor_pre

    @property
    def calibrated_post(self):
        """
        Returns the pressure p[n] calibrated with the reference post recording
        """
        return self._recording*self._cal_factor_post

    @property
    def calibration_factor_pre(self):
        """
        Returns the calibration factor that can be used to transform the digital
        values to actual pressure values in Pascal. This is factor is generated
        by the reference that was generated pre-recording.
        """
        return self._cal_factor_pre

    @property
    def calibration_factor_post(self):
        """
        Returns the calibration factor that can be used to transform the digital
        values to actual pressure values in Pascal. This is factor is generated
        by the reference that was generated post-recording.
        """
        return self._cal_factor_post


def root_mean_square(
    sampled_data,
):
    """
    Computes digital root mean square
    """
    return np.sqrt(np.sum(sampled_data.astype(np.float64)**2)/len(sampled_data))

def sound_pressure_level(
    pressure_rms,
    pressure_reference=Recording.pressure_reference,
):
    """
    Computes the sound pressure level (SPL)
    """
    return 20*np.log10(pressure_rms/pressure_reference)


def calibration_factor(
    digital_ref_rms: float,
    physical_ref_rms: float,  # in Pa
):
    """
    Multiply sampled signal with this factor to get values in Pascal
    """
    return physical_ref_rms/digital_ref_rms


def generate_short_series(
    sampled_data,
    time_length,
    sample_frequency,
):
    """
    Return an array with a length that corresponds to the given
    time_length with regard to the samplerate.
    """
    return sampled_data[0:int(sample_frequency*time_length)-1]


def power_spectrum(
    sampled_data,
):
    """
    Computes the power spectrum with forward scaling
    """
    return np.abs(
        np.fft.fft(
            a=sampled_data,
            n=len(sampled_data),
            norm="forward",
        ),
    )**2


def power_time_domain(
    sampled_data,
):
    """
    Computes the power from time/sample domain signal
    """
    return np.sum(sampled_data**2)


def power_frequency_domain(
    power_spectrum,
):
    """
    Computes the power from forward scaled power spectrum, which entails
    multiplying with the scaling factor due to |X(k)|^2
    """
    return np.sum(power_spectrum)*len(power_spectrum)


def sum_spl(
    spls: np.array,
):
    """
    Summing up of Sound Pressure Levels
    """
    return 10*np.log10(np.sum(10**(spls/10)))


def power_spectrum_to_db(
    power_spectrum,
    pressure_reference=Recording.pressure_reference,
):
    """
    Transforms a Power Spectrum with Pascal units to dB Spectrum
    """
    amplitude_spectrum = np.sqrt(power_spectrum)
    return 20*np.log10(amplitude_spectrum/pressure_reference)


def combine_power_spectrum(
    power_spectrum,  # dB
    power_spectrum_frequencies,
):
    """
    Combines constant bandwidth spectrum to 1Hz bands.
    """
    positive_power_spectrum_frequencies = power_spectrum_frequencies[0:len(power_spectrum_frequencies)//2-1]
    power_spectrum_combined = [power_spectrum[0]]
    power_spectrum_combined_frequencies = np.arange(0,24000)

    lower_index = 1
    for combined_freq in power_spectrum_combined_frequencies[1:]:
        temp_spls = []
        for index, freq in enumerate(positive_power_spectrum_frequencies[lower_index:]):
            if freq < combined_freq:
                temp_spls.append(power_spectrum[lower_index + index])
            else:
                lower_index = lower_index + index
                break
        power_spectrum_combined.append(sum_spl(np.array([temp_spls])))

    power_spectrum_combined = np.array(power_spectrum_combined) + 3  # add 3dB for adding negative spectrum

    return power_spectrum_combined_frequencies, power_spectrum_combined


def read_third_octave_band_csv(
    csv_path,
):
    """
    Read the 3rd octave band frequencies from the csv file and outputs
    a list with e.g.
        [{'f_lower': 10.0, 'f_mid': 12.5, 'f_upper': 14.0}, {...]
    """
    frequencies = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frequencies.append({
                'f_lower': float(row['f_lower']),
                'f_mid': float(row['f_mid']),
                'f_upper': float(row['f_upper']),
            })
    return frequencies


def read_a_weights_csv(
    csv_path,
):
    """
    Read A-weighted values from csv file and returning it as a dictionary
    with mid frequency as str as key and weight as value, e.g.
        {'12.5': -20.3, ... }
    """
    a_weights = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a_weights[str(float(row['frequency']))] = float(row['weight'])

    return a_weights


def generate_third_octave(
    power_spectrum_db,
    power_spectrum_frequencies,
    third_octave_frequencies,
):
    """
    Transforms a constant bandwidth power spectrum to a 3rd octave
    band spectrum
    """
    # Generate Indexes to fetch from power spectrum
    lower_index = 0
    indexes = []
    for third_octave_index, frequencies in enumerate(third_octave_frequencies):
        temp_index = {'third_octave_index': 0, 'lower_index': 0, 'upper_index': 0}
        for pow_spec_index, pow_frequency in enumerate(power_spectrum_frequencies[lower_index:]):
            if temp_index['lower_index'] == 0 and pow_frequency > frequencies['f_lower']:
                temp_index['third_octave_index'] = third_octave_index
                temp_index['lower_index'] = pow_spec_index + lower_index

            if temp_index['upper_index'] == 0 and pow_frequency > frequencies['f_upper']:
                temp_index['upper_index'] = pow_spec_index + lower_index - 1
                lower_index = pow_spec_index + lower_index - 1
                indexes.append(temp_index)
                break
    logging.debug(f"Lowest Index used: {indexes[0]['lower_index']}")
    logging.debug(f"Highest Index used: {indexes[-1]['upper_index']}")

    # Combine ranges in power spectrum to form third octave band
    third_octave_spectrum = []
    for index in indexes:
        positive_spl = sum_spl(
            spls=power_spectrum_db[index['lower_index']:index['upper_index']],
        )
        negative_spl = positive_spl
        third_octave_spectrum.append(
            sum_spl(
                spls=np.array([positive_spl, negative_spl]),
            )
        )

    return np.array(third_octave_spectrum)


def add_a_weighting(
    third_octave_db,
    third_octave_frequencies,
    a_weights,
):
    """
    Adds A-weighting to a 3rd octave band power spectrum.
    """
    a_weighted_spectrum = []
    for index, freq in enumerate(third_octave_frequencies):
        a_weighted_spectrum.append(
            10*np.log10(10**((third_octave_db[index]+a_weights[str(freq)])/10))
        )

    return np.array(a_weighted_spectrum)


def generate_plot(
    power_spectrum_db,  # dB
    power_spectrum_frequencies,
    third_octave_spectrum,  # dB
    a_weighted_third_octave_spectrum,  # dB
    third_octave_spectrum_frequencies,
):
    """
    Generate the plot for FFT, 3rd Octave band, and A-weighted
    3rd Octave band, similar to Figure 2 in L9
    """
    fig, ax = plt.subplots(1,1)

    freqs, spectrum = combine_power_spectrum(
        power_spectrum=power_spectrum_db,
        power_spectrum_frequencies=power_spectrum_frequencies,
    )

    ax.plot(freqs, spectrum, label="FFT dB")
    ax.step(third_octave_spectrum_frequencies, third_octave_spectrum, where="mid", label="1/3 dB")
    ax.step(third_octave_spectrum_frequencies, a_weighted_third_octave_spectrum, where="mid", label="1/3 dBA")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative SPL [dB]")
    ax.set_xlim(1,24000)
    ax.legend()
    ax.grid()
    logging.info("------------------ Plotting -----------------")
    logging.info("Close plot to continue")
    plt.show()


def calculate_short_sequence_spl(
    calibrated_recording,
    time_length,
    samplerate,
):
    """
    Calculate the Sound Pressure Level for a short sequence with time
    constant given by time_length.
    """
    cal_pressure = generate_short_series(
        sampled_data=calibrated_recording,
        time_length=time_length,
        sample_frequency=samplerate,
    )
    return sound_pressure_level(
        pressure_rms=root_mean_square(
            sampled_data=cal_pressure,
        ),
    )


def compute_values(
    calibrated_recording: np.array,
    samplerate: int,
    plot=False,
):
    """
    Calculate the values of interest and outputs them via logger.
    """
    # Calculate SPL
    logging.info("-------------- SPL from RMS -----------------")
    cal_pressure_rms_pre = root_mean_square(calibrated_recording)
    spl_rms = sound_pressure_level(pressure_rms=cal_pressure_rms_pre)
    logging.info(f"SPL: {spl_rms:.3f} dB")

    # Calculate Power Spectrum - Calibrated with Pre-Recording
    power_spectrum_pa = power_spectrum(calibrated_recording)
    power_spectrum_frequencies = np.fft.fftfreq(
        d=1/samplerate,
        n=len(power_spectrum_pa),
    )

    power_in_time_domain = power_time_domain(calibrated_recording)
    power_in_freq_domain = power_frequency_domain(
        power_spectrum=power_spectrum_pa,
    )

    logging.info("---- Power Values for Parseval's Theorem ----")
    logging.info(f"Power in Time Domain: {power_time_domain(calibrated_recording):.3f}")
    logging.info(f"Power in Frequency Domain: {power_in_freq_domain:.3f}")

    # Calculate SPL from Frequency Domain
    spl_freq = sum_spl(power_spectrum_to_db(power_spectrum_pa))
    logging.info("--- SPL from Constant Bandwidth Spectrum ----")
    logging.info(f"SPL: {spl_freq:.3f} dB")

    # Calculate Third Octave Band
    third_octave_frequencies = read_third_octave_band_csv(
        csv_path=third_octave_csv,
    )
    third_octave_mid_frequencies = [f['f_mid'] for f in third_octave_frequencies]
    third_octave_spectrum = generate_third_octave(
        power_spectrum_db=power_spectrum_to_db(power_spectrum_pa),
        power_spectrum_frequencies=power_spectrum_frequencies,
        third_octave_frequencies=third_octave_frequencies,
    )
    logging.info("--------- SPL from 3rd Octave Bands ---------")
    logging.info(f"SPL: {sum_spl(third_octave_spectrum):.3f} dB")

    # Calculate A-Weighted Third Octave Band
    a_weights = read_a_weights_csv(
        csv_path=a_weights_csv,
    )

    third_octave_a_weighted_spectrum = add_a_weighting(
        third_octave_db=third_octave_spectrum,
        third_octave_frequencies=third_octave_mid_frequencies,
        a_weights=a_weights,
    )
    third_octave_a_weighted_spl = sum_spl(third_octave_a_weighted_spectrum)
    logging.info("---- SPL from A-weighted 3rd Octave Bands ---")
    logging.info(f"SPL: {sum_spl(third_octave_a_weighted_spl):.3f} dB")

    if plot:
        generate_plot(
            power_spectrum_db=power_spectrum_to_db(power_spectrum_pa),  # dB
            power_spectrum_frequencies=power_spectrum_frequencies,
            third_octave_spectrum=third_octave_spectrum,  # dB
            a_weighted_third_octave_spectrum=third_octave_a_weighted_spectrum,  # dB
            third_octave_spectrum_frequencies=third_octave_mid_frequencies,
        )

    logging.info("=================== Done ====================")


def main():
    recording = Recording.from_recordings(
        recording_path=samfundet_rec_path,
        reference_pre_recording_path=samfundet_ref_pre_path,
        reference_post_recording_path=samfundet_ref_post_path,
    )
    logging.info("==== Values for Calibration Pre-Recording ===")

    compute_values(
        calibrated_recording=recording.calibrated_pre,
        samplerate=recording.samplerate,
        plot=False,
    )

    logging.info("=== Values for Calibration Post-Recording ===")

    compute_values(
        calibrated_recording=recording.calibrated_post,
        samplerate=recording.samplerate,
        plot=False,
    )

    logging.info("======= Values for Short Sequences ========")
    # Calculate SPL for 1s Sequence
    logging.info("----------- SPL for 1s Sequence -----------")
    spl_pre_short_1s = calculate_short_sequence_spl(
        calibrated_recording=recording.calibrated_pre,
        time_length=1,
        samplerate=recording.samplerate,
    )
    logging.info(f"SPL 1s Pre: {spl_pre_short_1s:.3f} dB")

    spl_post_short_1s = calculate_short_sequence_spl(
        calibrated_recording=recording.calibrated_post,
        time_length=1,
        samplerate=recording.samplerate,
    )
    logging.info(f"SPL 1s Post: {spl_post_short_1s:.3f} dB")

    # Calculate SPL for 0.125s Sequence
    logging.info("------------ SPL 125ms Sequence -------------")
    spl_pre_short_125ms = calculate_short_sequence_spl(
        calibrated_recording=recording.calibrated_pre,
        time_length=0.125,
        samplerate=recording.samplerate,
    )
    logging.info(f"SPL 0.125s Pre: {spl_pre_short_125ms:.3f} dB")

    spl_post_short_125ms = calculate_short_sequence_spl(
        calibrated_recording=recording.calibrated_post,
        time_length=0.125,
        samplerate=recording.samplerate,
    )
    logging.info(f"SPL 0.125s Post: {spl_post_short_125ms:.3f} dB")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

