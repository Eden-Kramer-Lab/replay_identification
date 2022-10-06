import copy
from logging import getLogger
from os.path import abspath, dirname, join, pardir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loren_frank_data_processing import (
    Animal,
    get_all_multiunit_indicators,
    get_all_spike_indicators,
    get_interpolated_position_dataframe,
    get_LFPs,
    get_spikes_dataframe,
    get_trial_time,
    make_neuron_dataframe,
    make_tetrode_dataframe,
)
from loren_frank_data_processing.position import make_track_graph
from ripple_detection import (
    Kay_ripple_detector,
    filter_ripple_band,
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from spectral_connectivity import Connectivity, Multitaper
from statsmodels.tsa.tsatools import lagmat

logger = getLogger(__name__)


SAMPLING_FREQUENCY = 500
_MARKS = ["channel_1_max", "channel_2_max", "channel_3_max", "channel_4_max"]
BRAIN_AREAS = ["CA1", "CA2", "CA3"]

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, "Raw-Data")

ANIMALS = {
    "bon": Animal(directory=join(RAW_DATA_DIR, "Bond"), short_name="bon"),
}

logger = getLogger(__name__)

_MARKS = ["channel_1_max", "channel_2_max", "channel_3_max", "channel_4_max"]


def get_labels(times, time):
    ripple_labels = pd.DataFrame(
        np.zeros_like(time, dtype=np.int), index=time, columns=["replay_number"]
    )
    for replay_number, start_time, end_time in times.itertuples():
        ripple_labels.loc[start_time:end_time] = replay_number

    return ripple_labels


def estimate_ripple_band_power(lfps, sampling_frequency):
    m = Multitaper(
        lfps.values,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.020,
        time_window_step=0.020,
        start_time=lfps.index[0].total_seconds(),
    )
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze() + np.spacing(1)
    n_samples = int(0.020 * sampling_frequency)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]
    power = pd.DataFrame(power, index=index)
    return power.reindex(lfps.index)


def get_adhoc_ripple(epoch_key, tetrode_info, position_time):
    LFP_SAMPLING_FREQUENCY = 1500
    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS).dropna(
        subset=["linear_position", "speed"]
    )
    speed = position_info["speed"]
    time = position_info.index

    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[(tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS)
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    ripple_filtered_lfps = pd.DataFrame(
        np.stack(
            [
                filter_ripple_band(ripple_lfps.values[:, ind], sampling_frequency=1500)
                for ind in np.arange(ripple_lfps.shape[1])
            ],
            axis=1,
        ),
        index=ripple_lfps.index,
    )
    ripple_times = Kay_ripple_detector(
        time,
        ripple_lfps.values,
        speed.values,
        LFP_SAMPLING_FREQUENCY,
        zscore_threshold=2.0,
        close_ripple_threshold=np.timedelta64(0, "ms"),
        minimum_duration=np.timedelta64(15, "ms"),
    )

    ripple_times.index = ripple_times.index.rename("replay_number")
    ripple_labels = get_labels(ripple_times, position_time)
    is_ripple = ripple_labels > 0
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds()
    )

    ripple_power = estimate_ripple_band_power(ripple_lfps, LFP_SAMPLING_FREQUENCY)
    interpolated_ripple_power = ripple_power.interpolate()

    ripple_power_change = interpolated_ripple_power.transform(lambda df: df / df.mean())
    ripple_power_zscore = np.log(interpolated_ripple_power).transform(
        lambda df: (df - df.mean()) / df.std()
    )

    return dict(
        ripple_times=ripple_times,
        ripple_labels=ripple_labels,
        ripple_filtered_lfps=ripple_filtered_lfps,
        ripple_power=ripple_power,
        ripple_lfps=ripple_lfps,
        ripple_power_change=ripple_power_change,
        ripple_power_zscore=ripple_power_zscore,
        is_ripple=is_ripple,
    )


def get_adhoc_multiunit(speed, tetrode_info, time_function):
    time = time_function()
    is_brain_areas = tetrode_info.area.astype(str).str.upper().isin(BRAIN_AREAS)
    tetrode_info = tetrode_info.loc[is_brain_areas]

    multiunit = (
        get_all_multiunit_indicators(tetrode_info.index, ANIMALS, time_function)
        .sel(features=_MARKS)
        .reindex({"time": time})
    )
    multiunit_spikes = (np.any(~np.isnan(multiunit.values), axis=1)).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, SAMPLING_FREQUENCY),
        index=time,
        columns=["firing_rate"],
    )
    multiunit_rate_change = multiunit_firing_rate.transform(lambda df: df / df.mean())
    multiunit_rate_zscore = np.log(multiunit_firing_rate).transform(
        lambda df: (df - df.mean()) / df.std()
    )

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time,
        multiunit_spikes,
        speed.values,
        SAMPLING_FREQUENCY,
        minimum_duration=np.timedelta64(15, "ms"),
        zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, "ms"),
    )
    multiunit_high_synchrony_times.index = multiunit_high_synchrony_times.index.rename(
        "replay_number"
    )
    multiunit_high_synchrony_labels = get_labels(multiunit_high_synchrony_times, time)
    is_multiunit_high_synchrony = multiunit_high_synchrony_labels > 0
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds()
    )

    return dict(
        multiunit=multiunit,
        multiunit_spikes=multiunit_spikes,
        multiunit_firing_rate=multiunit_firing_rate,
        multiunit_high_synchrony_times=multiunit_high_synchrony_times,
        multiunit_high_synchrony_labels=multiunit_high_synchrony_labels,
        multiunit_rate_change=multiunit_rate_change,
        multiunit_rate_zscore=multiunit_rate_zscore,
        is_multiunit_high_synchrony=is_multiunit_high_synchrony,
    )


def get_spikes(neuron_info, time_function):
    time = time_function()
    neuron_info = neuron_info.loc[
        (neuron_info.numspikes > 100)
        & neuron_info.area.isin(BRAIN_AREAS)
        & (neuron_info.type == "principal")
    ]
    spikes = get_all_spike_indicators(
        neuron_info.index, ANIMALS, time_function
    ).reindex(time)
    spike_times = [
        get_spikes_dataframe(neuron_key, ANIMALS) for neuron_key in neuron_info.index
    ]

    return spikes, spike_times


def load_data(epoch_key):
    logger.info("Loading data...")
    time = get_trial_time(epoch_key, ANIMALS)
    time = (
        pd.Series(np.ones_like(time, dtype=np.float), index=time)
        .resample("2ms")
        .mean()
        .index
    )

    def _time_function(*args, **kwargs):
        return time

    position_info = get_interpolated_position_dataframe(
        epoch_key, ANIMALS, _time_function
    ).dropna(subset=["linear_position", "speed"])

    time = position_info.index
    speed = position_info["speed"]

    neuron_info = make_neuron_dataframe(ANIMALS).xs(epoch_key, drop_level=False)
    spikes, spike_times = get_spikes(neuron_info, _time_function)
    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)

    track_graph, _ = make_track_graph(epoch_key, ANIMALS)

    logger.info("Finding multiunit high synchrony events...")
    adhoc_multiunit = get_adhoc_multiunit(speed, tetrode_info, _time_function)

    logger.info("Finding ripple times...")
    adhoc_ripple = get_adhoc_ripple(epoch_key, tetrode_info, time)
    return {
        "position_info": position_info,
        "tetrode_info": tetrode_info,
        "neuron_info": neuron_info,
        "spikes": spikes,
        "spike_times": spike_times,
        "track_graph": track_graph,
        "sampling_frequency": SAMPLING_FREQUENCY,
        **adhoc_ripple,
        **adhoc_multiunit,
    }


def plot_detector_debug(
    time_ind, data, replay_detector, detector_results, figsize=(10, 7.5)
):
    fig, axes = plt.subplots(
        7,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 3, 3, 3, 1, 1]},
    )

    place_fields = (
        replay_detector._spiking_likelihood.keywords["place_conditional_intensity"]
    ).squeeze()
    time = data["spikes"].iloc[time_ind].index / np.timedelta64(1, "s")

    place_field_max = np.argmax(place_fields, axis=0)
    linear_position_order = place_field_max.argsort(axis=0).squeeze()
    spike_time_ind, neuron_ind = np.nonzero(
        np.asarray(data["spikes"].iloc[time_ind])[:, linear_position_order]
    )

    axes[0].scatter(
        time[spike_time_ind], neuron_ind, clip_on=False, s=10, color="black", marker="|"
    )
    axes[0].set_ylim((0, place_fields.shape[1]))
    axes[0].set_yticks((0, place_fields.shape[1]))
    axes[0].set_ylabel("Cells")
    axes[0].fill_between(
        time,
        np.ones_like(time) * place_fields.shape[1],
        where=detector_results.isel(time=time_ind).replay_probability >= 0.8,
        color="lightgrey",
        zorder=-1,
        alpha=0.6,
        step="pre",
    )
    axes[0].fill_between(
        time,
        np.ones_like(time) * place_fields.shape[1],
        where=data["ripple_labels"].iloc[time_ind].values.squeeze() > 0,
        color="red",
        zorder=-2,
        alpha=0.1,
        step="pre",
    )

    detector_results.causal_posterior.isel(time=time_ind).sum("position").sel(
        state="Non-Local"
    ).plot(
        x="time", ax=axes[1], color="black", clip_on=False, linewidth=0.5, alpha=0.75
    )
    detector_results.isel(time=time_ind).replay_probability.plot(
        x="time", ax=axes[1], color="black", clip_on=False
    )
    axes[1].set_ylabel("Prob.")
    axes[1].set_xlabel("")
    axes[1].set_ylim((0, 1))

    cmap = copy.copy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey", alpha=1.0)
    (
        detector_results.isel(time=time_ind)
        .causal_posterior.sum("state")
        .where(replay_detector.is_track_interior_)
        .plot(x="time", y="position", ax=axes[2], cmap=cmap, vmin=0.0, vmax=0.02)
    )
    axes[2].scatter(
        data["position_info"].iloc[time_ind].index / np.timedelta64(1, "s"),
        data["position_info"].iloc[time_ind].linear_position,
        s=1,
        color="magenta",
        clip_on=False,
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Position")

    (
        detector_results.isel(time=time_ind)
        .acausal_posterior.sum("state")
        .where(replay_detector.is_track_interior_)
        .plot(x="time", y="position", ax=axes[3], cmap=cmap, vmin=0.0, vmax=0.02)
    )
    axes[3].scatter(
        data["position_info"].iloc[time_ind].index / np.timedelta64(1, "s"),
        data["position_info"].iloc[time_ind].linear_position,
        s=1,
        color="magenta",
        clip_on=False,
    )
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Position")

    (
        detector_results.isel(time=time_ind, state=1)
        .likelihood.where(replay_detector.is_track_interior_)
        .plot(x="time", y="position", ax=axes[4], vmin=0.75, vmax=1.0, cmap="viridis")
    )
    axes[4].scatter(
        data["position_info"].iloc[time_ind].index / np.timedelta64(1, "s"),
        data["position_info"].iloc[time_ind].linear_position,
        s=1,
        color="magenta",
        clip_on=False,
    )
    axes[4].set_xlabel("")
    axes[4].set_title("")
    axes[4].set_ylabel("Position")

    speed = np.asarray(data["position_info"].iloc[time_ind].speed).squeeze()
    axes[5].fill_between(time, speed, color="grey")
    axes[5].axhline(4, color="black", linestyle="--", linewidth=1)
    axes[5].set_ylabel("Speed")

    lagged_speed = np.asarray(
        lagmat(data["position_info"].speed, maxlag=1, use_pandas=True).iloc[time_ind]
    ).squeeze()
    probablity_replay = replay_detector.replay_state_transition_(lagged_speed)
    axes[6].plot(time, probablity_replay[:, 0])
    axes[6].twinx().plot(time, probablity_replay[:, 1], color="orange")

    sns.despine(offset=5)


def plot_detector(
    time_ind, data, replay_detector, detector_results, figsize=(10.5, 6.0)
):
    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 3, 1]},
    )

    place_fields = (
        replay_detector._spiking_likelihood.keywords["place_conditional_intensity"]
    ).squeeze()
    time = data["spikes"].iloc[time_ind].index / np.timedelta64(1, "s")

    place_field_max = np.argmax(place_fields, axis=0)
    linear_position_order = place_field_max.argsort(axis=0).squeeze()
    spike_time_ind, neuron_ind = np.nonzero(
        np.asarray(data["spikes"].iloc[time_ind])[:, linear_position_order]
    )

    axes[0].scatter(
        time[spike_time_ind], neuron_ind, clip_on=False, s=10, color="black", marker="|"
    )
    axes[0].set_ylim((0, place_fields.shape[1]))
    axes[0].set_yticks((0, place_fields.shape[1]))
    axes[0].set_ylabel("Cells")
    axes[0].fill_between(
        time,
        np.ones_like(time) * place_fields.shape[1],
        where=detector_results.isel(time=time_ind).replay_probability >= 0.8,
        color="lightgrey",
        zorder=-1,
        alpha=0.6,
        step="pre",
    )
    axes[0].fill_between(
        time,
        np.ones_like(time) * place_fields.shape[1],
        where=data["ripple_labels"].iloc[time_ind].values.squeeze() > 0,
        color="red",
        zorder=-2,
        alpha=0.1,
        step="pre",
    )

    detector_results.isel(time=time_ind).replay_probability.plot(
        x="time", ax=axes[1], color="black", clip_on=False
    )
    axes[1].set_ylabel("Prob.")
    axes[1].set_xlabel("")
    axes[1].set_ylim((0, 1))

    cmap = copy.copy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    (
        detector_results.isel(time=time_ind)
        .acausal_posterior.sum("state")
        .where(replay_detector.is_track_interior_)
        .plot(x="time", y="position", ax=axes[2], cmap=cmap, vmin=0.0, vmax=0.02)
    )
    axes[2].scatter(
        data["position_info"].iloc[time_ind].index / np.timedelta64(1, "s"),
        data["position_info"].iloc[time_ind].linear_position,
        s=1,
        color="magenta",
        clip_on=False,
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Position")

    speed = np.asarray(data["position_info"].iloc[time_ind].speed).squeeze()
    axes[3].fill_between(time, speed, color="grey")
    axes[3].axhline(4, color="black", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Speed")

    sns.despine(offset=5)
