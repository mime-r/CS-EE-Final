import warnings
warnings.filterwarnings('ignore')
import os
import glob
import csv
import librosa
import statistics

# Variable Settings
n_mfcc_var = 20


# Change the current working directory
try:
    os.chdir('./Final')
except:
    pass

# Part 1: Get Audio Files
my_path = "./Data/genres_original"
files = glob.glob(my_path + '/**/*.wav', recursive=True)

# Part 0: Initialise Feature File

f = open('./Data/original_30.csv', 'w', newline='')

# create the csv writer
writer = csv.writer(f)

header = []

def construct_header():
    mfcc_types = ["mean", "var"]
    header.append("filename")
    header.append("length")
    for i in range(n_mfcc_var):
        header.append(f"mfcc{str(i+1)}_{mfcc_types[0]}")
        header.append(f"mfcc{str(i+1)}_{mfcc_types[2]}")
    header.append("label")
    return

construct_header()

# write header to the csv file
writer.writerow(header)

# extract and write features
for file in files:
    filename = file.split('/')[-1]
    print(filename)
    try:
        y, sr = librosa.load(file)
    except:
        print("Error: ", filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_var)
    mfccs_mean = []
    for mfcc in mfccs:
        mfccs_mean.append(statistics.mean(mfcc))
    mfccs_var = []
    for mfcc in mfccs:
        mfccs_var.append(statistics.variance(mfcc))
    row = [filename.split("\\")[-1], len(y), *mfccs_mean, *mfccs_var, filename.split('.')[0].split("\\")[-1]]
    writer.writerow(row)


# close the file
f.close()

"""

x, sr = librosa.load(files[0])
mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc_var)
mfccs_mean = []
for mfcc in mfccs:
    mfccs_mean.append(statistics.mean(mfcc))

mfccs_variance = []
for mfcc in mfccs:
    mfccs_variance.append(statistics.variance(mfcc))
print(mfccs_mean)
print(mfccs_variance)
"""