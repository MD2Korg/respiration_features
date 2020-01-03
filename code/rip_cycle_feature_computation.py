# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:19:15 2018

@author: Nazir Saleheen, Md Azim Ullah
"""

from typing import List
import numpy as np

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
                           neighbor_ratio_stretch_datastream[:,2].reshape(-1,1)])
