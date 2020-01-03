# Copyright (c) 2018, MD2K Center of Excellence
# All rights reserved.
# author : Md Kauser Ahmmed
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List
from enum import Enum
import numpy as np
from copy import deepcopy

# TODO: What is this?
class Quality(Enum):
    ACCEPTABLE = 1
    UNACCEPTABLE = 0


def smooth(data:np.ndarray,
           span: int = 5)-> np.ndarray:
    """

    :rtype: object
    :param data:
    :param span:
    :return:
    """
    if data is None or len(data) == 0:
        return []
    # plt.figure()
    # plt.plot(data[:,1])
    sample = data[:,1]
    sample_middle = np.convolve(sample, np.ones(span, dtype=int), 'valid') / span
    divisor = np.arange(1, span - 1, 2)
    sample_start = np.cumsum(sample[:span - 1])[::2] / divisor
    sample_end = (np.cumsum(sample[:-span:-1])[::2] / divisor)[::-1]
    sample_smooth = np.concatenate((sample_start, sample_middle, sample_end))
    data[:,1] = sample_smooth

    # plt.plot(data[:,1])
    # plt.show()
    return data


def moving_average_curve(data: np.ndarray,
                         window_length: int) -> np.ndarray:
    """
    Moving average curve from filtered (using moving average) samples.

    :rtype: object
    :return: mac
    :param data:
    :param window_length:
    """
    if data is None or len(data) == 0:
        return []
    # plt.figure()
    # plt.plot(data[:,1])
    sample = data[:,1]
    for i in range(window_length, len(sample) - (window_length + 1)):
        sample_avg = np.mean(sample[i - window_length:i + window_length + 1])
        data[i,1] = sample_avg

    # plt.plot(data[:,1])
    # plt.show()
    return data[np.array(range(window_length, len(sample) - (window_length + 1)))]


def up_down_intercepts(data: np.ndarray,
                       mac: np.ndarray,
                       data_start_time_to_index: dict) -> [np.ndarray, np.ndarray]:
    """
    Returns Up and Down Intercepts.
    Moving Average Centerline curve intersects breath cycle twice. Once in the inhalation branch
    (Up intercept) and in the exhalation branch (Down intercept).

    :param data_start_time_to_index:
    :rtype: object
    :return up_intercepts, down_intercepts:
    :param data:
    :param mac:
    """

    up_intercepts = []
    down_intercepts = []

    subsets = []
    for i in range(len(mac)):
        data_index = data_start_time_to_index[mac[i,0]]
        subsets.append(data[data_index,1])
    # plt.plot(data[:,0],data[:,1])
    # plt.plot(mac[:,0],mac[:,1])
    # # plt.plot(mac[:,0],subsets)
    # plt.show()
    if len(subsets) == len(mac):
        for i in range(len(mac) - 1):
            if subsets[i] <= mac[i,1] and mac[i + 1,1] <= subsets[i + 1]:
                up_intercepts.append(mac[i + 1])
            elif subsets[i] >= mac[i,1] and mac[i + 1,1] >= subsets[i + 1]:
                down_intercepts.append(mac[i + 1])
    else:
        raise Exception("Data sample not found at Moving Average Curve.")

    return up_intercepts, down_intercepts

def filter_intercept_outlier(up_intercepts: List,
                             down_intercepts: List) -> [List,List]:
    """
    Remove two or more consecutive up or down intercepts.

    :rtype: object
    :return up_intercepts_updated, down_intercepts_updated:
    :param up_intercepts:
    :param down_intercepts:
    """

    up_intercepts_filtered = []
    down_intercepts_filtered = []

    for index in range(len(down_intercepts) - 1):
        up_intercepts_between_down_intercepts = []
        for ui in up_intercepts:
            if down_intercepts[index][0] <= ui[0] <= down_intercepts[index + 1][0]:
                up_intercepts_between_down_intercepts.append(ui)

        if len(up_intercepts_between_down_intercepts) > 0:
            up_intercepts_filtered.append(up_intercepts_between_down_intercepts[-1])

    up_intercepts_after_down = [ui for ui in up_intercepts if ui[0] > down_intercepts[-1][0]]
    if len(up_intercepts_after_down) > 0:
        up_intercepts_filtered.append(up_intercepts_after_down[-1])

    down_intercepts_before_up = [di for di in down_intercepts if di[0] < up_intercepts_filtered[0][0]]
    if len(down_intercepts_before_up) > 0:
        down_intercepts_filtered.append(down_intercepts_before_up[-1])

    for index in range(len(up_intercepts_filtered) - 1):
        down_intercepts_between_up_intercepts = []
        for di in down_intercepts:
            if up_intercepts_filtered[index][0] <= di[0] <= up_intercepts_filtered[index + 1][0]:
                down_intercepts_between_up_intercepts.append(di)

        if len(down_intercepts_between_up_intercepts) > 0:
            down_intercepts_filtered.append(down_intercepts_between_up_intercepts[-1])

    down_intercepts_after_up = [di for di in down_intercepts if di[0] > up_intercepts_filtered[-1][0]]
    if len(down_intercepts_after_up) > 0:
        down_intercepts_filtered.append(down_intercepts_after_up[-1])

    up_intercepts_truncated = []
    for ui in up_intercepts_filtered:
        if ui[0] >= down_intercepts_filtered[0][0]:
            up_intercepts_truncated.append(ui)

    min_length = min(len(up_intercepts_truncated), len(down_intercepts_filtered))

    up_intercepts_updated = up_intercepts_truncated[:min_length]
    down_intercepts_updated = down_intercepts_filtered[:min_length]

    return up_intercepts_updated, down_intercepts_updated



def generate_peak_valley(up_intercepts: List,
                         down_intercepts: List,
                         data: np.ndarray) -> [List, List]:
    """
    Compute peak valley from up intercepts and down intercepts indices.

    :rtype: object
    :return peaks, valleys:
    :param up_intercepts:
    :param down_intercepts:
    :param data:
    """
    peaks = []
    valleys = []

    last_iterated_index = 0
    for i in range(len(down_intercepts) - 1):
        peak = None
        valley = None

        for j in range(last_iterated_index, len(data)):
            if down_intercepts[i][0] <= data[j][0] <= up_intercepts[i][0]:
                if valley is None or data[j][1] < valley[1]:
                    valley = data[j]
            elif up_intercepts[i][0] <= data[j][0] <= down_intercepts[i + 1][0]:
                if peak is None or data[j][1] > peak[1]:
                    peak = data[j]
            elif data[j][0] > down_intercepts[i + 1][0]:
                last_iterated_index = j
                break

        valleys.append(valley)
        peaks.append(peak)

    return peaks, valleys


def correct_valley_position(peaks: List,
                            valleys: List,
                            up_intercepts: List,
                            data: np.ndarray,
                            data_start_time_to_index: dict) -> List:
    """
    Correct Valley position by locating actual valley using maximum slope algorithm which is
    located between current valley and following peak.

    Algorithm - push valley towards right:
    Search for points lies in between current valley and following Up intercept.
    Calculate slopes at those points.
    Ensure that valley resides at the begining of inhalation cycle where inhalation slope is maximum.

    :rtype: object
    :return valley_updated:
    :param peaks:
    :param valleys:
    :param up_intercepts:
    :param data:
    :param data_start_time_to_index: hash table for data where start_time is key, index is value
    """
    valley_updated = valleys.copy()
    for i in range(len(valleys)):
        if valleys[i][0] < up_intercepts[i][0] < peaks[i][0]:
            up_intercept = up_intercepts[i]

            if valleys[i][0] not in data_start_time_to_index or up_intercept[0] not in data_start_time_to_index:
                exception_message = 'Data has no start time for valley or up intercept start time at index ' + str(i)
                raise Exception(exception_message)
            else:
                valley_index = data_start_time_to_index[valleys[i][0]]
                up_intercept_index = data_start_time_to_index[up_intercept[0]]
                data_valley_to_ui = data[valley_index: up_intercept_index + 1]
                sample_valley_to_ui = data_valley_to_ui[:,1]

                slope_at_samples = np.diff(sample_valley_to_ui)

                consecutive_positive_slopes = [-1] * len(slope_at_samples)

                for j in range(len(slope_at_samples)):
                    slopes_subset = slope_at_samples[j:]
                    if all(slope > 0 for slope in slopes_subset):
                        consecutive_positive_slopes[j] = len(slopes_subset)

                if any(no_con_slope > 0 for no_con_slope in consecutive_positive_slopes):
                    indices_max_pos_slope = []
                    for k in range(len(consecutive_positive_slopes)):
                        if consecutive_positive_slopes[k] == max(consecutive_positive_slopes):
                            indices_max_pos_slope.append(k)
                    valley_updated[i] = data_valley_to_ui[indices_max_pos_slope[-1]]

        else:
            # TODO: discuss whether raise exception or not
            # Up intercept at index i is not between valley and peak at index i.
            break

    return valley_updated


def correct_peak_position(peaks: List,
                          valleys: List,
                          up_intercepts: List,
                          data: np.ndarray,
                          max_amplitude_change_peak_correction: float,
                          min_neg_slope_count_peak_correction: int,
                          data_start_time_to_index: dict) -> List:
    """
    Correct peak position by checking if there is a notch in the inspiration branch at left position.
    If at least 60% inspiration is done at a notch point, assume that notch as an original peak.
    Our hypothesis is most of breathing in done for that cycle. We assume insignificant amount of
    breath is taken or some new cycle started after the notch.

    :rtype: object
    :return peaks:
    :param peaks:
    :param valleys:
    :param up_intercepts:
    :param data:
    :param max_amplitude_change_peak_correction:
    :param min_neg_slope_count_peak_correction:
    :param data_start_time_to_index: hash table for data where start_time is key, index is value

    """
    for i, item in enumerate(peaks):
        if valleys[i][0] < up_intercepts[i][0] < peaks[i][0]:
            up_intercept = up_intercepts[i]
            # points between current valley and UI.
            if up_intercept[0] not in data_start_time_to_index or peaks[i][0] not in data_start_time_to_index:
                exception_message = 'Data has no start time for peak or up intercept start time at index ' + str(i)
                raise Exception(exception_message)
            else:
                data_up_intercept_index = data_start_time_to_index[up_intercept[0]]
                data_peak_index = data_start_time_to_index[peaks[i][0]]

                data_ui_to_peak = data[data_up_intercept_index: data_peak_index + 1]

                sample_ui_to_peak = data_ui_to_peak[:,1]
                slope_at_samples = np.diff(sample_ui_to_peak)

                if not all(j >= 0 for j in slope_at_samples):
                    indices_neg_slope = [j for j in range(len(slope_at_samples)) if slope_at_samples[j] < 0]
                    peak_new = data_ui_to_peak[indices_neg_slope[0]]
                    valley_peak_dist_new = peak_new[1] - valleys[i][1]
                    valley_peak_dist_prev = peaks[i][1] - valleys[i][1]
                    if valley_peak_dist_new == 0:
                        raise Exception("New peak to valley distance is equal to zero. "
                                        "This will encounter divide by zero exception.")
                    else:
                        amplitude_change = (valley_peak_dist_prev - valley_peak_dist_new) / valley_peak_dist_new * 100.0

                        if len(indices_neg_slope) >= min_neg_slope_count_peak_correction:
                            if amplitude_change <= max_amplitude_change_peak_correction:
                                peaks[i] = peak_new  # 60% inspiration is done at that point.

        else:
            # TODO: Discuss whether raise exception or not for this scenario.
            break  # up intercept at i is not between valley and peak at i

    return peaks


def remove_close_valley_peak_pair(peaks: List,
                                  valleys: List,
                                  minimum_peak_to_valley_time_diff: float = 0.31) -> [List, List]:
    """
    Filter out too close valley peak pair.

    :rtype: object
    :return peaks_updated, valleys_updated:
    :param peaks:
    :param valleys:
    :param minimum_peak_to_valley_time_diff:
    """

    peaks_updated = []
    valleys_updated = []

    for i, item in enumerate(peaks):
        time_diff_valley_peak = peaks[i][0] - valleys[i][0]
        if time_diff_valley_peak/1000 > minimum_peak_to_valley_time_diff:
            peaks_updated.append(peaks[i])
            valleys_updated.append(valleys[i])

    return peaks_updated, valleys_updated

def filter_expiration_duration_outlier(peaks: List,
                                       valleys: List,
                                       threshold_expiration_duration: float) -> [List, List]:
    """
    Filter out peak valley pair for which expiration duration is too small.

    :rtype: object
    :return peaks_updated, valleys_updated:
    :param peaks:
    :param valleys:
    :param threshold_expiration_duration:
    """

    peaks_updated = []
    valleys_updated = [valleys[0]]

    for i, item in enumerate(peaks):
        if i < len(peaks) - 1:
            expiration_duration = valleys[i + 1][0] - peaks[i][0]
            if expiration_duration/1000 > threshold_expiration_duration:
                peaks_updated.append(peaks[i])
                valleys_updated.append(valleys[i + 1])

    peaks_updated.append(peaks[-1])

    return peaks_updated, valleys_updated


def filter_small_amp_expiration_peak_valley(peaks: List,
                                            valleys: List,
                                            expiration_amplitude_threshold_perc: float) -> [List, List]:
    """
    Filter out peak valley pair if their expiration amplitude is less than or equal to 10% of
    average expiration amplitude.

    :rtype: object
    :return: peaks_updated, valleys_updated:
    :param: peaks:
    :param: valleys:
    :param: expiration_amplitude_threshold_perc:
    """

    expiration_amplitudes = []
    peaks_updated = []
    valleys_updated = [valleys[0]]

    for i, peak in enumerate(peaks):
        if i < len(peaks) - 1:
            expiration_amplitudes.append(abs(valleys[i + 1][1] - peak[1]))

    mean_expiration_amplitude = np.mean(expiration_amplitudes)

    for i, expiration_amplitude in enumerate(expiration_amplitudes):
        if expiration_amplitude > expiration_amplitude_threshold_perc * mean_expiration_amplitude:
            peaks_updated.append(peaks[i])
            valleys_updated.append(valleys[i + 1])

    peaks_updated.append(peaks[-1])

    return peaks_updated, valleys_updated


def filter_small_amp_inspiration_peak_valley(peaks: List,
                                             valleys: List,
                                             inspiration_amplitude_threshold_perc: float) -> [List,List]:
    """
    Filter out peak valley pair if their inspiration amplitude is less than or to equal 10% of
    average inspiration amplitude.

    :rtype: object
    :return peaks_updated, valleys_updated:
    :param peaks:
    :param valleys:
    :param inspiration_amplitude_threshold_perc:
    """

    peaks_updated = []
    valleys_updated = []

    inspiration_amplitudes = [(peaks[i][1] - valleys[i][1]) for i, valley in enumerate(valleys)]
    mean_inspiration_amplitude = np.mean(inspiration_amplitudes)

    for i, inspiration_amplitude in enumerate(inspiration_amplitudes):
        if inspiration_amplitude > inspiration_amplitude_threshold_perc * mean_inspiration_amplitude:
            valleys_updated.append(valleys[i])
            peaks_updated.append(peaks[i])

    return peaks_updated, valleys_updated


def compute_peak_valley(rip: np.ndarray,
                        fs: float = 21.33,
                        smoothing_factor: int = 5,
                        time_window: int = 8,
                        expiration_amplitude_threshold_perc: float = 0.10,
                        threshold_expiration_duration: float = 0.312,
                        inspiration_amplitude_threshold_perc: float = 0.10,
                        max_amplitude_change_peak_correction: float = 30,
                        min_neg_slope_count_peak_correction: int = 4,
                        minimum_peak_to_valley_time_diff=0.31) -> [np.ndarray, np.ndarray]:
    """
    Compute peak and valley from rip data and filter peak and valley.

    :rtype: object
    :param minimum_peak_to_valley_time_diff:
    :param inspiration_amplitude_threshold_perc:
    :param smoothing_factor:
    :return peak_datastream, valley_datastream:
    :param rip:
    :param rip_quality:
    :param fs:
    :param time_window:
    :param expiration_amplitude_threshold_perc:
    :param threshold_expiration_duration:
    :param max_amplitude_change_peak_correction:
    :param min_neg_slope_count_peak_correction:
    """

    rip_filtered = rip
    data_smooth = smooth(data=rip_filtered, span=smoothing_factor)
    window_length = int(round(time_window * fs))
    # plt.figure()
    # plt.plot(rip_filtered[:,0],rip_filtered[:,1])
    # plt.plot(data_smooth[:,0],data_smooth[:,1])
    # # plt.plot(data_mac[:,0],data_mac[:,1])
    # plt.show()
    data_mac = moving_average_curve(deepcopy(data_smooth), window_length=window_length)

    data_smooth_start_time_to_index = {}
    for index, data in enumerate(data_smooth):
        data_smooth_start_time_to_index[data_smooth[index,0]] = index

    up_intercepts, down_intercepts = up_down_intercepts(data=data_smooth,
                                                        mac=data_mac,
                                                        data_start_time_to_index=data_smooth_start_time_to_index)
    # print(up_intercepts,down_intercepts)
    up_intercepts_filtered, down_intercepts_filtered = filter_intercept_outlier(up_intercepts=up_intercepts,
                                                                                down_intercepts=down_intercepts)

    peaks, valleys = generate_peak_valley(up_intercepts=up_intercepts_filtered,
                                          down_intercepts=down_intercepts_filtered,
                                          data=data_smooth)

    valleys_corrected = correct_valley_position(peaks=peaks,
                                                valleys=valleys,
                                                up_intercepts=up_intercepts_filtered,
                                                data=data_smooth,
                                                data_start_time_to_index=data_smooth_start_time_to_index)


    peaks_corrected = correct_peak_position(peaks=peaks,
                                            valleys=valleys_corrected,
                                            up_intercepts=up_intercepts_filtered,
                                            data=data_smooth,
                                            max_amplitude_change_peak_correction=max_amplitude_change_peak_correction,
                                            min_neg_slope_count_peak_correction=min_neg_slope_count_peak_correction,
                                            data_start_time_to_index=data_smooth_start_time_to_index)

    # remove too close valley peak pair.
    peaks_filtered_close, valleys_filtered_close = remove_close_valley_peak_pair(peaks=peaks_corrected,
                                                                                 valleys=valleys_corrected,
                                                                                 minimum_peak_to_valley_time_diff=minimum_peak_to_valley_time_diff)

    # Remove small  Expiration duration < 0.31
    peaks_filtered_exp_dur, valleys_filtered_exp_dur = filter_expiration_duration_outlier(peaks=peaks_filtered_close,
                                                                                          valleys=valleys_filtered_close,
                                                                                          threshold_expiration_duration=threshold_expiration_duration)

    # filter out peak valley pair of inspiration of small amplitude.
    peaks_filtered_insp_amp, valleys_filtered_insp_amp = filter_small_amp_inspiration_peak_valley(
        peaks=peaks_filtered_exp_dur,
        valleys=valleys_filtered_exp_dur,
        inspiration_amplitude_threshold_perc=inspiration_amplitude_threshold_perc)

    # filter out peak valley pair of expiration of small amplitude.
    peaks_filtered_exp_amp, valleys_filtered_exp_amp = filter_small_amp_expiration_peak_valley(
        peaks=peaks_filtered_insp_amp,
        valleys=valleys_filtered_insp_amp,
        expiration_amplitude_threshold_perc=expiration_amplitude_threshold_perc)

    peaks_filtered_exp_amp, valleys_filtered_exp_amp = np.array(peaks_filtered_exp_amp), np.array(valleys_filtered_exp_amp)
    # peaks_filtered_exp_amp = np.insert(peaks_filtered_exp_amp,
    # peaks_filtered_exp_amp =  np.insert(peaks_filtered_exp_amp, 2, 5, axis=1)
    # valleys_filtered_exp_amp =  np.insert(valleys_filtered_exp_amp, 2, 5, axis=1)
    # peaks_filtered_exp_amp[:,2] = [data_smooth_start_time_to_index[i[0]] for i in peaks_filtered_exp_amp]
    # valleys_filtered_exp_amp[:,2] = [data_smooth_start_time_to_index[i[0]] for i in valleys_filtered_exp_amp]
    return peaks_filtered_exp_amp, valleys_filtered_exp_amp

def rip_cycle_feature_computation(peaks_datastream: np.ndarray,
                                  valleys_datastream: np.ndarray) -> np.ndarray:
    """
    Respiration Feature Implementation. The respiration feature values are
    derived from the following paper:
    'puffMarker: a multi-sensor approach for pinpointing the timing of first lapse in smoking cessation'
    Removed due to lack of current use in the implementation
    roc_max = []  # 8. ROC_MAX = max(sample[j]-sample[j-1])
    roc_min = []  # 9. ROC_MIN = min(sample[j]-sample[j-1])

    :param peaks_datastream: list of peak datapoints
    :param valleys_datastream: list of valley datapoints
    :return: lists of DataPoints each representing a specific feature calculated from the respiration cycle
    found from the peak valley inputs
    """

    inspiration_duration = []  # 1 Inhalation duration
    expiration_duration = []  # 2 Exhalation duration
    respiration_duration = []  # 3 Respiration duration
    inspiration_expiration_ratio = []  # 4 Inhalation and Exhalation ratio
    stretch = []  # 5 Stretch
    upper_stretch = []  # 6. Upper portion of the stretch calculation
    lower_stretch = []  # 7. Lower portion of the stretch calculation
    delta_previous_inspiration_duration = []  # 10. BD_INSP = INSP(i)-INSP(i-1)
    delta_previous_expiration_duration = []  # 11. BD_EXPR = EXPR(i)-EXPR(i-1)
    delta_previous_respiration_duration = []  # 12. BD_RESP = RESP(i)-RESP(i-1)
    delta_previous_stretch_duration = []  # 14. BD_Stretch= Stretch(i)-Stretch(i-1)
    delta_next_inspiration_duration = []  # 19. FD_INSP = INSP(i)-INSP(i+1)
    delta_next_expiration_duration = []  # 20. FD_EXPR = EXPR(i)-EXPR(i+1)
    delta_next_respiration_duration = []  # 21. FD_RESP = RESP(i)-RESP(i+1)
    delta_next_stretch_duration = []  # 23. FD_Stretch= Stretch(i)-Stretch(i+1)
    neighbor_ratio_expiration_duration = []  # 29. D5_EXPR(i) = EXPR(i) / avg(EXPR(i-2)...EXPR(i+2))
    neighbor_ratio_stretch_duration = []  # 32. D5_Stretch = Stretch(i) / avg(Stretch(i-2)...Stretch(i+2))

    valleys = valleys_datastream
    peaks = peaks_datastream[:-1]

    for i, peak in enumerate(peaks):
        valley_start_time = valleys[i][0]
        valley_end_time = valleys[i + 1][0]

        delta = peak[0] - valleys[i][0]
        inspiration_duration.append(np.array([valley_start_time,valley_end_time,delta/1000]))

        delta = valleys[i + 1][0] - peak[0]
        expiration_duration.append(np.array([valley_start_time,valley_end_time,delta/1000]))

        delta = valleys[i + 1][0] - valley_start_time
        respiration_duration.append(np.array([valley_start_time,valley_end_time,delta/1000]))

        ratio = (peak[0] - valley_start_time) / (valleys[i + 1][0] - peak[0])
        inspiration_expiration_ratio.append(np.array([valley_start_time,valley_end_time,ratio]))

        value = peak[1] - valleys[i + 1][1]
        stretch.append(np.array([valley_start_time,valley_end_time,value]))

    for i, point in enumerate(inspiration_duration):
        valley_start_time = valleys[i][0]
        valley_end_time = valleys[i + 1][0]
        if i == 0:  # Edge case
            delta_previous_inspiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_previous_expiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_previous_respiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_previous_stretch_duration.append(np.array([valley_start_time,valley_end_time,0]))
        else:
            delta = inspiration_duration[i][2] - inspiration_duration[i - 1][2]
            delta_previous_inspiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = expiration_duration[i][2] - expiration_duration[i - 1][2]
            delta_previous_expiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = respiration_duration[i][2] - respiration_duration[i - 1][2]
            delta_previous_respiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = stretch[i][0] - stretch[i - 1][2]
            delta_previous_stretch_duration.append(np.array([valley_start_time,valley_end_time,delta]))

        if i == len(inspiration_duration) - 1:
            delta_next_inspiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_next_expiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_next_respiration_duration.append(np.array([valley_start_time,valley_end_time,0]))
            delta_next_stretch_duration.append(np.array([valley_start_time,valley_end_time,0]))
        else:
            delta = inspiration_duration[i][2] - inspiration_duration[i + 1][2]
            delta_next_inspiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = expiration_duration[i][2] - expiration_duration[i + 1][2]
            delta_next_expiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = respiration_duration[i][2] - respiration_duration[i + 1][2]
            delta_next_respiration_duration.append(np.array([valley_start_time,valley_end_time,delta]))

            delta = stretch[i][2] - stretch[i + 1][2]
            delta_next_stretch_duration.append(np.array([valley_start_time,valley_end_time,delta]))

        stretch_average = 0
        expiration_average = 0
        count = 0.0
        for j in [-2, -1, 1, 2]:
            if i + j < 0 or i + j >= len(inspiration_duration):
                continue
            stretch_average += stretch[i + j][2]
            expiration_average += expiration_duration[i + j][2]
            count += 1

        stretch_average /= count
        expiration_average /= count

        ratio = stretch[i][2] / stretch_average
        neighbor_ratio_stretch_duration.append(np.array([valley_start_time,valley_end_time,ratio]))

        ratio = expiration_duration[i][2] / expiration_average
        neighbor_ratio_expiration_duration.append(np.array([valley_start_time,valley_end_time,ratio]))

    # Begin assembling datastream for output
    inspiration_duration_datastream = np.array(inspiration_duration)[1:-1]

    expiration_duration_datastream = np.array(expiration_duration)[1:-1]

    respiration_duration_datastream = np.array(respiration_duration)[1:-1]

    inspiration_expiration_ratio_datastream = np.array(inspiration_expiration_ratio)[1:-1]

    stretch_datastream = np.array(stretch)[1:-1]

    delta_previous_inspiration_duration_datastream = np.array(delta_previous_inspiration_duration)[1:-1]

    delta_previous_expiration_duration_datastream = np.array(delta_previous_expiration_duration)[1:-1]

    delta_previous_respiration_duration_datastream = np.array(delta_previous_respiration_duration)[1:-1]

    delta_previous_stretch_duration_datastream = np.array(delta_previous_stretch_duration)[1:-1]

    delta_next_inspiration_duration_datastream = np.array(delta_next_inspiration_duration)[1:-1]

    delta_next_expiration_duration_datastream = np.array(delta_next_expiration_duration)[1:-1]

    delta_next_respiration_duration_datastream = np.array(delta_next_respiration_duration)[1:-1]

    delta_next_stretch_duration_datastream = np.array(delta_next_stretch_duration)[1:-1]

    neighbor_ratio_expiration_datastream = np.array(neighbor_ratio_expiration_duration)[1:-1]

    neighbor_ratio_stretch_datastream = np.array(neighbor_ratio_stretch_duration)[1:-1]

    return np.concatenate([inspiration_duration_datastream,
                           expiration_duration_datastream[:,2].reshape(-1,1),
                           respiration_duration_datastream[:,2].reshape(-1,1),
                           inspiration_expiration_ratio_datastream[:,2].reshape(-1,1),
                           stretch_datastream[:,2].reshape(-1,1),
                           delta_previous_inspiration_duration_datastream[:,2].reshape(-1,1),
                           delta_previous_expiration_duration_datastream[:,2].reshape(-1,1),
                           delta_previous_respiration_duration_datastream[:,2].reshape(-1,1),
                           delta_previous_stretch_duration_datastream[:,2].reshape(-1,1),
                           delta_next_inspiration_duration_datastream[:,2].reshape(-1,1),
                           delta_next_expiration_duration_datastream[:,2].reshape(-1,1),
                           delta_next_respiration_duration_datastream[:,2].reshape(-1,1),
                           delta_next_stretch_duration_datastream[:,2].reshape(-1,1),
                           neighbor_ratio_expiration_datastream[:,2].reshape(-1,1),
                           neighbor_ratio_stretch_datastream[:,2].reshape(-1,1)],axis=1)

import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('rip.csv',delimiter=',').values[100000:200000]
peaks,valleys = compute_peak_valley(data)
features = rip_cycle_feature_computation(peaks,valleys)
plt.plot(features[:,0],features[:,4])
plt.show()
print(features.shape)
plt.plot(data[:,0],data[:,1])
plt.plot(np.array(peaks)[:,0],np.array(peaks)[:,1],'*')
plt.plot(np.array(valleys)[:,0],np.array(valleys)[:,1],'o')
plt.show()











