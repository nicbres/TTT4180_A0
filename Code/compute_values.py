#!./venv/bin/python3
import csv
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

logging.basicConfig(level=logging.INFO)

ntnu_dir_path = pathlib.Path('/home') / 'nbresina' / 'Documents' / 'NTNU'
recordings_path = ntnu_dir_path / 'TechnicalAcoustics' / 'Assignments' / '0' / 'Recordings'

samfundet_folder_path = recordings_path / 'Samfundet'
samfundet_ref_pre_path = samfundet_folder_path / 'ZOOM0004_reference_pre' / 'ZOOM0004_Tr1.WAV'
#samfundet_rec_path = samfundet_folder_path / 'ZOOM0007_samfundet' / 'ZOOM0007_Tr1.WAV'
samfundet_rec_path = samfundet_folder_path / 'ZOOM0007_samfundet' / 'ZOOM0007_Noise_Rec_30min.wav'
samfundet_ref_post_path = samfundet_folder_path / 'ZOOM0008_reference_post' / 'ZOOM0008_Tr1.WAV'

#construction_folder_path = recordings_path / 'ConstructionSite'
#construction_ref_pre_path = construction_folder_path / 'ZOOM0009_reference_pre' / 'ZOOM0009_Tr1.WAV'
#construction_rec_path = construction_folder_path / 'ZOOM0010_construction_site' / 'ZOOM0010_Tr1.WAV'
#construction_ref_post_path = construction_folder_path / 'ZOOM0011_reference_post' / 'ZOOM0011_Tr1.WAV'

reference_pressure = 20*10**(-6)

code_path = ntnu_dir_path / 'TechnicalAcoustics' / 'Assignments' / '0' / 'Code'
a_weights_csv = code_path / 'a_weights.csv'
third_octave_csv = code_path / 'third_octave_frequencies.csv'


class Recording:
    def __init__(
        self,
        ref_pre_path: pathlib.Path,
        rec_data_path: pathlib.Path,
        ref_post_path: pathlib.Path,
    ):
        logging.info("------------- Reading WAV files -------------")
        samplerate, ref_pre = wav.read(ref_pre_path)
        logging.info(f"ref_pre samplerate: {samplerate}")
        samplerate,  ref_post = wav.read(ref_post_path)
        logging.info(f"ref_post samplerate: {samplerate}")
        samplerate, rec_data = wav.read(rec_data_path)
        logging.info(f"rec samplerate: {samplerate}")
        logging.info("------------------- Done -------------------")
        self._samplerate = samplerate
        self._ref_pre = ref_pre
        self._ref_post = ref_post
        self._rec_data = rec_data

    @property
    def reference_pre(self):
        return self._ref_pre

    @property
    def reference_post(self):
        return self._ref_post

    @property
    def recording(self):
        return self._rec_data

    @property
    def samplerate(self):
        return self._samplerate


def root_mean_square(
    sampled_data,
):
    return np.sqrt(np.sum(sampled_data.astype(np.float64)**2)/len(sampled_data))

def sound_pressure_level(
    pressure_rms,
):
    return 20*np.log10(pressure_rms/reference_pressure)


def calculate_pressure(
    digital_rec_rms,
    digital_ref_rms,
    calibrator_pressure,
):
    return digital_rec_rms/digital_ref_rms*calibrator_pressure


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
    return sampled_data[0:int(sample_frequency*time_length)-1]


def power_spectrum(
    sampled_data,
):
    return np.abs(
        np.fft.fft(
            a=sampled_data,
            n=len(sampled_data),
        ),
    )**2


def power_time_domain(
    sampled_data,
):
    return np.sum(sampled_data**2)


def power_frequency_domain(
    power_spectrum,
):
    return np.sum(power_spectrum)/len(power_spectrum)


def sum_spl(
    spls: np.array,
):
    return 10*np.log10(np.sum(10**(spls/10)))


def power_spectrum_to_db(
    power_spectrum,
):
    amplitude_spectrum = np.sqrt(power_spectrum)/len(power_spectrum)
    return 20*np.log10(amplitude_spectrum/reference_pressure)


def read_csv(
    csv_path,
):
    csv_list = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_list.append(row)
    return csv_list 


def read_third_octave_band_csv(
    csv_path,
):
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
    third_octave_spl = []
    for index in indexes:
        positive_spl = sum_spl(
            spls=power_spectrum_db[index['lower_index']:index['upper_index']],
        )
        negative_spl = positive_spl
        third_octave_spl.append(
            sum_spl(
                spls=np.array([positive_spl, negative_spl]),
            )
        )

    return np.array(third_octave_spl)


def add_a_weighting(
    third_octave_db,
    third_octave_frequencies,
    a_weights,
):
    a_weighted_spectrum = []
    for index, freq in enumerate(third_octave_frequencies):
        a_weighted_spectrum.append(
            10*np.log10(10**((third_octave_db[index]+a_weights[str(freq)])/10))
        )

    return np.array(a_weighted_spectrum)


if __name__ == '__main__':
    samfundet_recording = Recording(
        ref_pre_path=samfundet_ref_pre_path,
        rec_data_path=samfundet_rec_path,
        ref_post_path=samfundet_ref_post_path,
    )

    # Calculate pressure on calibrator
    calibrator_pressure = 10**(94/20)*reference_pressure
    logging.info(f"Calibrator pressure [Pa]: {calibrator_pressure:.4f}")

    # Calculate calibration factor
    logging.info("-------------- Digital RMS ----------------")
    digital_ref_rms_pre = root_mean_square(
        sampled_data=samfundet_recording.reference_pre,
    )
    logging.info(f"Digital RMS Pre: {digital_ref_rms_pre:.4f}")
    digital_ref_rms_post = root_mean_square(
        sampled_data=samfundet_recording.reference_post,
    )
    logging.info(f"Digital RMS Post: {digital_ref_rms_post:.4f}")

    # Calibrate measured signal
    cal_factor_pre = calibration_factor(
        digital_ref_rms=digital_ref_rms_pre,
        physical_ref_rms=calibrator_pressure,
    )
    cal_pressure_pre = cal_factor_pre*samfundet_recording.recording

    cal_factor_post = calibration_factor(
        digital_ref_rms=digital_ref_rms_post,
        physical_ref_rms=calibrator_pressure,
    )
    cal_pressure_post = cal_factor_post*samfundet_recording.recording

    # Calculate calibrated RMS pressures
    logging.info("---------- Calibrated Pressure RMS ----------")
    cal_pressure_rms_pre = root_mean_square(cal_pressure_pre)
    logging.info(
        f"Pressure RMS calibrated with Pre: {cal_pressure_rms_pre:.4f} Pa",
    )

    cal_pressure_rms_post = root_mean_square(cal_pressure_post)
    logging.info(
        f"Pressure RMS calibrated with Post: {cal_pressure_rms_post:.4f} Pa",
    )

    # Calculate SPL
    logging.info("------------ SPL -------------")
    spl_pre = sound_pressure_level(
        pressure_rms=cal_pressure_rms_pre,
    )
    logging.info(
        f"SPL Rec with Pre: {spl_pre:.4f} dB",
    )
    spl_post = sound_pressure_level(
        pressure_rms=cal_pressure_rms_post,
    )
    logging.info(
        f"SPL Rec with Post: {spl_post:.4f} dB",
    )

    # Calculate SPL for 1s Sequence
    logging.info("--------- SPL 1s Sequence ---------")
    cal_pressure_pre_short_1s = generate_short_series(
        sampled_data=cal_pressure_pre,
        time_length=1,
        sample_frequency=samfundet_recording.samplerate,
    )
    spl_pre_short_1s = sound_pressure_level(
        pressure_rms=root_mean_square(
            sampled_data=cal_pressure_pre_short_1s,
        )
    )
    logging.info(f"SPL calibrated with Pre: {spl_pre_short_1s:.4f} dB")

    # Calculate SPL for 0.125s Sequence
    logging.info("--------- SPL 125ms Sequence ---------")
    cal_pressure_pre_short_125ms = generate_short_series(
        sampled_data=cal_pressure_pre,
        time_length=0.125,
        sample_frequency=samfundet_recording.samplerate,
    )
    spl_pre_short_125ms = sound_pressure_level(
        pressure_rms=root_mean_square(
            sampled_data=cal_pressure_pre_short_125ms,
        )
    )
    logging.info(f"SPL calibrated with Pre: {spl_pre_short_125ms:.4f} dB")

    # Generate Windowed Signals
    """
    cal_pressure_pre = generate_short_series(
        sampled_data=cal_pressure_pre,
        time_length=60,
        sample_frequency=samfundet_recording.samplerate,
    )
    logging.info("--------- SPL 60s Sequence ---------")
    spl_pre_short_60s = sound_pressure_level(
        pressure_rms=root_mean_square(
            sampled_data=cal_pressure_pre,
        )
    )
    logging.info(f"SPL calibrated with Pre: {spl_pre_short_60s:.4f} dB")
    """
    window = sig.windows.get_window(
        window=('kaiser', 4.0),
        Nx=len(cal_pressure_pre),
        fftbins=False,
    )
    windowed_cal_pressure_pre = window*cal_pressure_pre
    """
    spl_pre_short_60s_windowed = sound_pressure_level(
        pressure_rms=root_mean_square(
            sampled_data=windowed_cal_pressure_pre,
        )
    )
    logging.info(f"SPL windowed calibrated with Pre: {spl_pre_short_60s_windowed:.4f} dB")
    """

    # Calculate FFT
    power_spectrum_pre = power_spectrum(cal_pressure_pre)
    power_spectrum_pre_windowed = power_spectrum(windowed_cal_pressure_pre)

    power_spectrum_pre_freq = np.fft.fftfreq(
        d=1/samfundet_recording.samplerate,
        n=len(power_spectrum_pre),
    )

    power_time_domain_ = power_time_domain(cal_pressure_pre)
    power_time_domain_windowed = power_time_domain(windowed_cal_pressure_pre)
    power_freq_domain = power_frequency_domain(
        power_spectrum=power_spectrum_pre,
    )
    power_freq_domain_windowed = power_frequency_domain(
        power_spectrum=power_spectrum_pre_windowed,
    )

    logging.info("------------ Power Values -------------")
    logging.info(f"Power in Time Domain: {power_time_domain_:.4f}")
    logging.info(f"Power in Time Domain Windowed: {power_time_domain_windowed:.4f}")
    logging.info(f"Power in Frequency Domain: {power_freq_domain:.4f}")
    logging.info(f"Power in Frequency Domain Windowed: {power_freq_domain_windowed:.4f}")

    # Calculate SPL from Frequency Domain
    spl_freq = sum_spl(power_spectrum_to_db(power_spectrum_pre))
    spl_freq_windowed = sum_spl(power_spectrum_to_db(power_spectrum_pre_windowed))
    logging.info("------- SPL from Frequency Domain --------")
    logging.info(f"SPL: {spl_freq:.4f} dB")
    logging.info(f"SPL windowed: {spl_freq_windowed:.4f} dB")

    # Calculate Third Octave Band
    third_octave_frequencies = read_third_octave_band_csv(
        csv_path=third_octave_csv,
    )
    third_octave_mid_frequencies = [f['f_mid'] for f in third_octave_frequencies]
    third_octave_spl = generate_third_octave(
        power_spectrum_db=power_spectrum_to_db(power_spectrum_pre),
        power_spectrum_frequencies=power_spectrum_pre_freq,
        third_octave_frequencies=third_octave_frequencies,
    )
    logging.info("------- SPL from Third Octave Bands --------")
    logging.info(f"SPL: {sum_spl(third_octave_spl):.4f} dB")

    # Calculate A-Weighted Third Octave Band
    a_weights = read_a_weights_csv(
        csv_path=a_weights_csv,
    )

    third_octave_a_weighted_spectrum = add_a_weighting(
        third_octave_db=third_octave_spl,
        third_octave_frequencies=third_octave_mid_frequencies,
        a_weights=a_weights,
    )
    third_octave_a_weighted_spl_pos = sum_spl(third_octave_a_weighted_spectrum)
    third_octave_a_weighted_spl = sum_spl(np.array([
        third_octave_a_weighted_spl_pos,
        third_octave_a_weighted_spl_pos,
    ]))
    logging.info("------- A-weighted SPL from Third Octave Bands --------")
    logging.info(f"SPL: {sum_spl(third_octave_a_weighted_spl):.4f} dB")


"""
    # Plotting
    fig, axs = plt.subplots(3,1)
    #axs[0].plot(power_spectrum_pre_freq, power_spectrum_to_db(power_spectrum_pre))
    #axs[0].plot(power_spectrum_pre_freq, power_spectrum_pre)
    #axs[0].set_yscale("log")
    #axs[0].set_xscale("log")
    axs[2].plot(third_octave_mid_frequencies, third_octave_a_weighted_spectrum, 'o')
    axs[2].set_yscale("log")
    axs[2].set_xscale("log")
    axs[1].plot(third_octave_mid_frequencies, third_octave_spl, 'o')
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    plt.show()
"""

