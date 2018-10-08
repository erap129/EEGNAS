import numpy as np
import mne
from scipy.io import loadmat

class BCI_IV_2a():
    def __init__(self, filename, labels_filename=None):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info['events'] = events
        cnt.info['artifact_trial_mask'] = artifact_trial_mask
        return cnt

    def extract_data(self):
        raw_edf = mne.io.read_raw_edf(self.filename, stim_channel='auto')
        raw_edf.load_data()
        data = raw_edf.get_data()

        # this bit of code turns all of the minimum values in each channel to the mean value (w/o the minimum)
        # why do they do this? I don't know
        for i_chan in range(data.shape[0] - 1):
            this_chan = data[i_chan]
            data[i_chan] = np.where(this_chan == np.min(this_chan),
                                    np.nan, this_chan)
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean
        gdf_events = raw_edf.find_edf_events()
        raw_edf = mne.io.RawArray(data, raw_edf.info, verbose='WARNING')
        raw_edf.info['gdf_events'] = gdf_events
        return raw_edf

    def extract_events(self, raw_edf):
        # create a list of tuples (event_time, event_code)
        events = np.array(list(zip(
            raw_edf.info['gdf_events'][1],
            raw_edf.info['gdf_events'][2]
        )))
        trial_mask = [ev_code in [769, 770, 771, 772, 783]
                      for ev_code in events[:, 1]]
        trial_events = events[trial_mask]
        assert len(trial_events) == 288, (
            "Got {:d} markers".format(len(trial_events))
        )
        # turn 769 to 1, 770 to 2, and so on
        trial_events[:, 1] = trial_events[:, 1] - 768
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)['classlabel'].squeeze()
            trial_events[:, 1] = classes
        unique_classes = np.unique(trial_events[:, 1])
        assert np.array_equal([1,2,3,4], unique_classes), (
            "Expect 1,2,3,4 as class labels, got {:s}".format(
                str(unique_classes))
        )
        trial_start_events = events[events[:,1] == 768]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:,1] == 1023]
        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        events = np.zeros((len(trial_events), 3), dtype=np.int32)
        events[:, 0] = trial_events[:, 0]
        events[:, 2] = trial_events[:, 1]

        return events, artifact_trial_mask


