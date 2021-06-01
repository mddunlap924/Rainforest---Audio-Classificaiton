import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import librosa
import librosa.display
import numpy as np
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
import random
import scipy
import colorednoise as cn


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class TimeStretch:
    def __init__(self, max_rate=1.2):
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0.25, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented, rate


class PitchShift:
    def __init__(self, max_steps=5, sr=48000):
        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented


class PinkNoiseSNR:
    def __init__(self, min_snr=5.0, max_snr=20.0, **kwargs):
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


# Data Augmentation
def data_augmentation(y, label, sr):
    # # Pitch Shift
    # pitch_shift_transform = PitchShift(max_steps=2, sr=sr)
    # y_ = pitch_shift_transform.apply(y)

    # Time Stretch
    time_stretch_transform = TimeStretch(max_rate=2.0)
    y_, rate = time_stretch_transform.apply(y)

    # Pink Noise
    pn_transform = PinkNoiseSNR(min_snr=5.0, max_snr=20.0)
    y_ = pn_transform.apply(y_)

    return y_


def segment_split_cals(start, end, L, data_length, augment):
    # end = 2876673
    # start = 2725886
    # L = 96000
    # data_length = 2880000
    # augment = False

    # Basic Dimension Checks for Chunking Data
    D = end - start
    N = (D // L) + 1
    R = (L * N) - D
    offset = R // 2
    if augment:
        offset = np.random.randint(offset // 10, offset, 1)

    start_check = (start - offset) >= 0
    end_check = ((start - offset) + (L * N)) <= data_length

    segments = {}
    if start_check and end_check:
        for i in range(N):
            left = (start - offset) + (L * i)
            right = left + L
            idx_range = np.arange(start=left, stop=right, step=1, dtype=np.int)
            y = (idx_range >= start) & (idx_range <= end)
            segments[i] = {'y_true': y * 1.0,
                           'idx': idx_range
                           }
    else:
        if end_check:
            for i in range(N):
                right = data_length - (L * i)
                left = right - L
                idx_range = np.arange(start=left, stop=right, step=1, dtype=np.int)
                y = (idx_range >= start) & (idx_range <= end)
                segments[i] = {'y_true': y * 1.0,
                               'idx': idx_range
                               }
        if start_check:
            for i in range(N):
                left = (L * i)
                right = left + L
                idx_range = np.arange(start=left, stop=right, step=1, dtype=np.int)
                y = (idx_range >= start) & (idx_range <= end)
                segments[i] = {'y_true': y * 1.0,
                               'idx': idx_range
                               }

    # # Plot Results
    # for i, _ in enumerate(segments):
    #     idx_start, idx_end = segments[i]['idx'][0], segments[i]['idx'][-1]
    #     plt.figure()
    #     plt.plot(segments[i]['y_true'])
    #     plt.title(f'Sample {i + 1} of {len(segments)}: {idx_start} to {idx_end}; {idx_end - idx_start + 1}')
    #     plt.show()

    return segments


def segment_split_cals_center(start, end, L, data_length, augment):
    # end = 2876673
    # start = 2725886
    # L = 96000
    # data_length = 2880000
    # augment = False


    # Basic Dimension Checks for Chunking Data
    # Positioning sound slice
    center = np.round((start + end) / 2)
    beginning = center - L / 2
    if beginning < 0:
        beginning = 0

    # if augment:
    #     seed_everything(7)
    #     beginning = np.random.randint(beginning, center)

    ending = beginning + L
    if ending > data_length:
        ending = data_length
    beginning = ending - L

    N = int(np.ceil((end - start) / L) + 1)
    idx = []
    for i in range(N):
        if i == 0:
            idx.append([int(beginning), int(ending)])
        else:
            # Shift a window to the left of the center
            left_left = int((beginning - (L * i)))
            left_right = int(left_left + L)
            # Check that some of the left shifted image is still in start and end range
            if left_right > start:
                if left_left >= 0:
                    idx.append([left_left, left_right])

            # Shift a window to the right of the center
            right_left = int((beginning + (L * i)))
            right_right = int(right_left + L)
            # Check that some of the right shifted image is still in start and end range
            if right_left < end:
                if right_right <= data_length:
                    idx.append([right_left, right_right])

    segments = {}
    for i, idx_ in enumerate(idx):
        idx_range = np.arange(start=idx_[0], stop=idx_[1], step=1, dtype=np.int)
        y = (idx_range >= start) & (idx_range <= end)

        segments[i] = {'y_true': y * 1.0,
                       'idx': idx_range
                       }

    return segments


def segment_no_split_cals(start, end, L, data_length, augment):
    # Positioning sound slice
    center = np.round((start + end) / 2)
    beginning = center - L / 2
    if beginning < 0:
        beginning = 0

    if augment:
        seed_everything(7)
    else:
        seed_everything(42)
    beginning = np.random.randint(beginning, center)
    ending = beginning + L
    if ending > data_length:
        ending = data_length
    beginning = ending - L

    idx_range = np.arange(start=beginning, stop=ending, step=1, dtype=np.int)
    y = (idx_range >= start) & (idx_range <= end)

    segment = {0: {'y_true': y * 1.0,
                   'idx': idx_range
                   }}

    return segment


class Dataset:
    def __init__(self, name, data_type, img_size, process, save, *, augment=False):
        self.name = name
        self.data_type = data_type
        self.process = process
        self.save = save
        self.__directory = os.getcwd()
        self.clip_duration = float()
        self.fft = int()
        self.hop = int()
        self.power = int()
        self.sr = int()
        self.length = float()
        self.fmin = float()
        self.fmax = float()
        self.n_mels = int()
        self.resize = img_size
        self.X = np.array([])
        self.Y = np.array([])
        self.file_names = []
        self.species = []
        self.file_rows = []
        self.augment = augment
        self.__y_label = np.array([])
        self.overlap = float()
        # self.__tmin = float()
        # self.__tmax = float()

        print(f'Dataset on Disk: {self.__dataset_saved_on_disk()}')

    # Check if the dataset is already saved on disk/uploaded to kaggle
    def __dataset_saved_on_disk(self):
        current_dir = self.__directory
        dataset_path = os.path.join(current_dir, os.path.join('Dataset', self.name))
        dataset_exist = os.path.isfile(dataset_path)

        return dataset_exist

    # List TP training files
    def __train_tp_files(self):
        csv_path = os.path.join(self.__directory, 'rfcx-species-audio-detection\\train_tp.csv')
        data_path = os.path.join(self.__directory, 'rfcx-species-audio-detection\\train')

        with open(os.path.join(csv_path)) as f:
            reader = csv.reader(f)
            data = list(reader)

        return data

    # Prepare training dataset for processing
    def __freq_time_range(self):

        # TP training csv file
        data = self.__train_tp_files()

        # Check minimum/maximum frequencies for bird calls
        # Not necessary, but there are usually plenty of noise in low frequencies, and removing it helps
        fmin = int(self.sr / 2)
        fmax = 0

        # Skip header row (recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max) and start from 1 instead of 0
        t_delta = []
        for i in range(1, len(data)):
            fmin_ = float(data[i][4])
            fmax_ = float(data[i][6])
            if fmin > fmin_:
                fmin = float(data[i][4])
            if fmax < fmax_:
                fmax = float(data[i][6])

            t_delta.append(float(data[i][5]) - float(data[i][3]))

        # Get some safety margin
        fmin = int(fmin * 0.9)
        fmax = int(fmax * 1.1)
        print('Training Data - Minimum Frequency: ' + str(fmin) + ', Maximum Frequency: ' + str(fmax))

        # Maximum time delta for any call
        t_delta_max = max(t_delta)
        print(f'Training Data - Maximum Time Duration for a Call {round(t_delta_max, 2)} seconds')

        self.fmin = fmin
        self.fmax = fmax

        return

    def alternative_mel_spectrogram(self, signal):
        fft = self.fft
        hop = self.hop
        sr = self.sr
        fmin = self.fmin
        fmax = self.fmax
        power = self.power
        n_mels = self.n_mels
        window = 'hann'

        # signal_ = signal
        # fft, hop, power, n_mels = 4096, 256, 3, 300

        if self.augment:
            signal = data_augmentation(signal, self._Dataset__y_label, self.sr)
            random_powers = [self.power]
            random_mels = [self.n_mels]
            random_fmin = [40, 50, 60, 70, 80]
            random_fmax = [20000, 22000, 24000]
            random_fft = [self.fft]
            random_hop = [self.hop]
            random_window = ['hann']
            power = random_powers[np.random.randint(0, len(random_powers))]
            n_mels = random_mels[np.random.randint(0, len(random_mels))]
            fmin = random_fmin[np.random.randint(0, len(random_fmin))]
            fmax = random_fmax[np.random.randint(0, len(random_fmax))]
            fft = random_fft[np.random.randint(0, len(random_fft))]
            hop = random_hop[np.random.randint(0, len(random_hop))]
            window = random_window[np.random.randint(0, len(random_window))]

        mel_filter = librosa.filters.mel(
            sr=sr,
            n_fft=fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        m = np.abs(librosa.core.stft(
            signal,
            n_fft=fft,
            hop_length=hop,
            # win_length=win_length,
            center=True,
            window=window,
            pad_mode='reflect',
            dtype=np.complex,
        )) ** power
        m = np.dot(mel_filter, m)
        m1 = librosa.core.power_to_db(m, ref=np.max).astype(np.float32)
        # m1 = skimage.transform.resize(m1, (self.resize[0], self.resize[1]), anti_aliasing=True)
        # fig, ax = plt.subplots(); im = ax.imshow(m1, aspect='auto'); fig.colorbar(im, ax=ax)

        mel_specs = {'power': power,
                     'n_mels': n_mels,
                     'fmin': fmin,
                     'fmax': fmax,
                     'hop': hop,
                     'mel_filter': mel_filter
                     }

        return m1, mel_specs

    # Split a test *.flac file
    def __split_test_data(self, test_file):

        wav, sr = librosa.load(test_file, sr=None)
        # print(f'SR {sr}')
        self.sr = sr
        length = sr * self.clip_duration
        self.length = length

        # Take segments at 1 second intervals
        segment_length = sr * 1
        num_segs = int(len(wav) / segment_length)
        idx_ranges = []
        for i in range(num_segs):
            idx_start = i * segment_length
            idx_end = idx_start + length
            if idx_end <= len(wav):
                idx_ranges.append([idx_start, idx_end])
                if idx_end == len(wav):
                    break

        # Mel Spectrogram
        slices = {}
        for i, idx_range in enumerate(idx_ranges):
            idx_start = idx_range[0]
            idx_end = idx_range[1]
            signal = wav[int(idx_start):int(idx_end)]
            mel_spec_db, mel_specs = self.alternative_mel_spectrogram(signal)
            mel_db_resize = skimage.transform.resize(mel_spec_db, (self.resize[0], self.resize[1]), anti_aliasing=True)
            mel_db_scaled = mel_db_resize
            slices[i] = {'idx_start': idx_start,
                         'idx_end': idx_end,
                         'signal': signal,
                         'mel_spec': mel_db_scaled,
                         'time_start': idx_start / sr,
                         'time_end': idx_end / sr}

        slices['wav_length'] = len(wav)

        return slices

    # Split and process a train *.flac file
    def __split_train_data(self, data, data_csv):

        data_path = os.path.join(self.__directory, 'rfcx-species-audio-detection\\train')
        data_file = os.path.join(data_path, data[0] + '.flac')

        wav, sr = librosa.load(data_file, sr=None)

        """ Y-label encoding for (Classes x time) """
        # y_label for entire sound clip
        # print(f'SR {sr}')
        self.sr = sr
        length = self.clip_duration * sr
        self.length = length

        t_min = int(float(data[3]) * sr)
        t_max = int(float(data[5]) * sr)

        # Check if window is longer/shorter than call time
        if (t_max - t_min) > length:
            if self.augment:
                segments = segment_split_cals(t_min, t_max, length, len(wav), self.augment)
            else:
                segments = segment_split_cals_center(t_min, t_max, length, len(wav), self.augment)
        else:
            segments = segment_no_split_cals(t_min, t_max, length, len(wav), self.augment)

        # Loop through each segment
        slices = []
        for segment_count in range(len(segments)):
            segment = segments[segment_count]
            beginning = segment['idx'][0]
            ending = segment['idx'][-1]
            y_label_clip = np.zeros([len(wav), 24], dtype=np.float32)
            # Check if other species are present in clipped segment
            recording_id = data[0]
            all_events = []
            for data_row in data_csv[1:]:
                if data_row[0] == recording_id:
                    t_min_ = int(float(data_row[3]) * sr)
                    t_max_ = int(float(data_row[5]) * sr)
                    # check_overlap = (t_min_ < ending) and (t_max_ > beginning)
                    check_overlap = (t_min_ < ending) or (t_max_ > beginning)
                    if check_overlap:
                        y_label_clip[t_min_:t_max_, int(data_row[1])] = 1.0
                        all_events.append(int(data_row[1]))

            all_events_unique = np.unique(np.array(all_events)).tolist()

            signal = wav[segment['idx']]
            y_label = y_label_clip[segment['idx']]
            self.__y_label = y_label

            # Mel Spectrogram
            mel_spec_db, mel_specs_ = self.alternative_mel_spectrogram(signal)

            # Resize spectogram and labels
            mel_db_resize = skimage.transform.resize(mel_spec_db, (self.resize[0], self.resize[1]), anti_aliasing=True)
            mel_db_scaled = mel_db_resize
            y_label_resize = scipy.signal.resample(y_label, self.resize[1], axis=0)
            y_label_resize[y_label_resize > 0.65] = 1.0
            y_label_resize[y_label_resize < 0.65] = 0.0

            # Skip header row (recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max) and start from 1 instead of 0
            slice = {'idx_start': int(beginning),
                     'idx_end': int(ending),
                     'signal': signal,
                     'mel_spec': mel_db_scaled,
                     'species_id': all_events_unique,
                     'recording_id': data[0],
                     'time_start': int(beginning) * sr,
                     'time_end': int(ending) * sr,
                     'y_label': y_label_resize}

            slices.append(slice)

        return slices

    # Process the data to make images
    def mel_images(self, parameters):
        self.clip_duration = parameters['clip_duration']
        self.fft = parameters['fft']
        self.hop = parameters['hop']
        self.power = parameters['power']
        self.n_mels = parameters['n_mels']
        self.overlap = parameters['overlap']
        # self.length = self.clip_duration * self.sr

        if 'fmin' in parameters:
            self.fmin = parameters['fmin']
            self.fmax = parameters['fmax']

        fft = self.fft
        hop = self.hop
        # Less rounding errors this way
        sr = self.sr
        length = self.clip_duration * sr
        self.length = length

        # Training Data Files
        if self.data_type == 'Train':
            if 'fmin' not in parameters:
                self.__freq_time_range()
            # Data files start on row 1 (not 0 - 0 is header info)
            tp_data = self.__train_tp_files()
            # X = np.zeros([1, self.resize[0], self.resize[1]])
            # Y = np.zeros([1, self.resize[1], 24])

            # Loop through each file and segment in a given file
            file_names = []
            species = []
            file_rows = []
            X = []
            Y = []
            count = 0
            for i, data_row in enumerate(tp_data[1:]):
                slices = self.__split_train_data(data_row, tp_data)
                for slice_ in slices:
                    X.append(slice_['mel_spec'])
                    Y.append(slice_['y_label'])
                    file_names.append(data_row[0])
                    species.append(slice_['species_id'])
                    file_rows.append(str(i))
                    count += 1
                if len(slices) > 1:
                    split_window = True
                else:
                    split_window = False
                print(f'Training File {i + 1} of {len(tp_data[1:])}; Split {split_window}; Count {count}')
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)

            self.X = X
            self.Y = Y
            self.file_names = file_names
            self.species = species

        # Test Data Files
        elif self.data_type == 'Test':
            data_path = os.path.join(self.__directory, 'rfcx-species-audio-detection\\test')
            test_files = os.listdir(data_path)
            slices = self.__split_test_data(os.path.join(data_path, test_files[0]))

            # Test dataset zero-padded numpy array
            X = np.zeros([len(test_files), len(slices) - 1, self.resize[0], self.resize[1]])

            # File Name
            file_names = []

            # Loop through all test files
            for i, test_file in enumerate(test_files):
                file_names.append(test_file)
                slices = self.__split_test_data(os.path.join(data_path, test_file))
                for j in range(len(slices) - 1):
                    X[i, j, :, :] = slices[j]['mel_spec']
                print(f'Test File {i + 1} of {len(test_files)}')

            self.file_names = file_names
            self.X = X
            self.Y = None

        return


if __name__ == '__main__':
    # Inputs
    DATA_TYPE = 'Training'  # ['Training', 'Testing', 'Both']
    AUGMENT_DATA = False
    mel_parameters = {'clip_duration': 2,
                      'fft': 2048,
                      'hop': 512,
                      'power': 2,
                      'fmin': 40,
                      'fmax': 24000,
                      'n_mels': 224,
                      'overlap': 0.5,
                      }
    dataset_name = 'Mel_224_224_2'
    data_folder = r'C:\Kaggle\RainForest_R0\Datasets\Mel_224_224_2'
    img_size = (224, 224)

    if DATA_TYPE == 'Training':
        if AUGMENT_DATA:
            file_name_extension = '_training_augment.pickle'
            seed_everything(42)
        else:
            file_name_extension = '_training.pickle'
            seed_everything(42)
        ds_train = Dataset(name=dataset_name, data_type='Train', img_size=img_size, process=True, save=True,
                           augment=AUGMENT_DATA)
        ds_train.mel_images(parameters=mel_parameters)
        with open(os.path.join(data_folder, dataset_name + file_name_extension), 'wb') as output:
            pickle.dump(ds_train, output, pickle.HIGHEST_PROTOCOL)

    if DATA_TYPE == 'Testing':
        seed_everything(seed=42)
        ds_test = Dataset(name=dataset_name, data_type='Test', img_size=img_size, process=True, save=True)
        ds_test.mel_images(parameters=mel_parameters)

        with open(os.path.join(data_folder, dataset_name + '_testing.pickle'), 'wb') as output:
            pickle.dump([ds_test, mel_parameters], output, pickle.HIGHEST_PROTOCOL)

    print('End of Script')
    print('Check Point')
