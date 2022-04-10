import scipy.signal
import numpy as np
from scipy.ndimage.filters import median_filter

# default: armband
# 1. Apply the butterworth high pass filter at 2Hz
# 2. shift_electrodes
def butter_highpass(cutoff, fs, order=3):
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# usage: butter_highpass_filter(channel_example, 2, 200)
# channel_example: (52,)
def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

# 每个被试的每个round/session的所有数据做shift
# session_x: [n, (8, 52)]
def get_final_shifting(session_x, session_y):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        cwt_add = [] # (8, 52)
        for j in range(len(session_x)): # (5311, 8, 52)
            if Y_example[j] == classe:
                if cwt_add == []:
                    cwt_add = np.array(session_x[j])
                else:
                    cwt_add += np.array(session_x[j])
        # 3,4,5,6类的所有样本对应通道对应帧求和，然后帧内所有通道求和，并找出最大的帧
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0))) # class_mean: [32, 26, 34, 33]

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10) # array([1, 4, 8, 1])
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10) # array([9, 6, 2, 9])

    shifts_array = [] # [-1, -4, 2, -1]
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    return final_shifting

# 一个round/session的所有数据做shift
def shift_electrodes(examples, labels): # [28, (189, 1, 8, 52)], [28, (189,)]
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k])

        cwt_add = [] # (8, 52)
        for j in range(len(X_example)): # (5311, 1, 8, 52)
            if Y_example[j] == classe:
                if cwt_add == []:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        # 3,4,5,6类的所有样本对应通道对应帧求和，然后帧内所有通道求和，并找出最大的帧
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0))) # class_mean: [32, 26, 34, 33]

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10) # array([1, 4, 8, 1])
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10) # array([9, 6, 2, 9])

    shifts_array = [] # [-1, -4, 2, -1]
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)): # [28, (189, 1, 8, 52)]
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting)) # final_shifting = -1
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example



# default: ninapro
# f: cut-off frequency  fs:sampling frequency
def lpf(x, f=1, fs=100):
    # return x
    wn = 2.0 *f / fs
    b, a = scipy.signal.butter(1, wn, 'low')
    x = np.abs(x)
    output = scipy.signal.filtfilt(
        b, a, x, axis=0,
        padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)
    )
    # 濾波之後可能索引倒排，需要copy
    return output.copy()

# default: cslhdemg
def bpf(x, order=4, cut_off=[20, 400], sampling_f = 2048):
    # return x
    wn = [2.0 * i / sampling_f for i in cut_off]
    b, a = scipy.signal.butter(order, wn, "bandpass")
    output = scipy.signal.filtfilt(b, a, x)
    # print(x.shape, output.shape)
    return output.copy()

def butter_bandpass_filter(x, order=4, lowcut=20, highcut=400, fs = 2048):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    y = scipy.signal.lfilter(b, a, x)
    return y


def butter_bandstop_filter(x, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
    y = scipy.signal.lfilter(b, a, x)
    return y


def butter_lowpass_filter(x, order=1, cut=1, fs=100, zero_phase=False):
    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = scipy.signal.butter(order, cut, btype='low')
    y = (scipy.signal.filtfilt if zero_phase else scipy.signal.lfilter)(b, a, x)
    return y


def amplify(x, rate=1000):
    return x * rate

def u_normalization(x, u):
    return np.sign(x) * np.log(1 + u*np.abs(x)) / np.log(1 + u)

def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1) # 找出每段的分界点
    # print(label, breaks)
    for begin, end in zip([0] + breaks, breaks + [len(label)]): # 构造段 [begin, end)
        assert begin < end
        yield begin, end

# for csl-hdemg:  x: (6144, 168)
def csl_cut(x):
    window = 150
    last = x.shape[0] // window * window
    # print(x[:last].shape)
    new_x = x[:last].reshape((-1, window, x[:last].shape[1])) # (40, 150, 168)
    # print(new_x.shape)
    rms = np.sqrt(np.mean(np.square(new_x), axis=1)) # (40, 168)
    # print(rms.shape)
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)] # (40, 168)
    # rms = [scipy.signal.medfilt(i, 3) for i in rms] # (40, 168)
    # print(len(rms), rms[0].shape)
    rms = np.mean(rms, axis=1) # (40,)
    threshold = np.mean(rms)
    mask = rms > threshold # (40,)
    # print(mask.shape)
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    begin, end = max(continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0])) # 比较规则，段起始是true，并且长度最长

    begin = begin * window
    end = end * window
    return x[begin:end]

def median_filt(x, order=3):  # x: (n, 168)
    return np.array([median_filter(image, 3).ravel() for image
                    in x.reshape(-1, 24, 7)]).astype(np.float32)


    new_x = x.reshape(-1, 24, 7)
    for i in range(new_x.shape[0]):
        new_x[i] = median_filter(new_x[i], 3)
    new_x = new_x.reshape(-1, 168)
    return new_x
    return scipy.signal.medfilt(x, (1,3))

def abs(x):
    return np.abs(x)

def downsample(x, rate):
    return x[::rate].copy()