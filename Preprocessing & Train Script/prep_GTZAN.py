
##Adapted from Yuan Gong (2021) -- yuangong@mit.edu 

import numpy as np
import json
import os
import wget
# from google.colab import drive
# drive.mount('/content/drive')
# PATH_OF_DATA= '/content/gdrive/MyDrive/GTZAN/Data/'


# prepare the data of the GTZAN dataset.
print('Now download and process GTZAN dataset, it will take a few moments...')
if os.path.exists('./data') == False:
    os.mkdir('./data')
# download the speechcommands dataset
if os.path.exists('./data/GTZAN') == False:
    # we use the 35 class v2 dataset, which is used in torchaudio https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html
    sc_url = 'https://www.dropbox.com/scl/fi/80mng4iut8k16urm58awr/GTZANshort.tar.gz?rlkey=w2hiud7zo92e3foli490rect4&st=mddvegxr&dl=1'
    wget.download(sc_url, out='./data/')
    print('Finished Download, Extracting')
    os.mkdir('./data/GTZAN')
    os.system('tar -xzvf ./data/GTZANshort.tar.gz -C ./data/GTZAN')
    #os.remove('./data/GTZAN.tar.gz')
# Use training list from coreyker - "https://github.com/coreyker/dnn-mgr/tree/master/gtzan"
with open('./data/GTZAN/Data/train_filtered.txt', 'r') as f:
  train_list = f.readlines()
  extra_list = []
  for line in train_list:
    newline = line.strip()
    parts = line.split('/')
    parts[0] += '2'
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
    parts[0].replace('2', '3')
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
  train_list += extra_list
print("Train_list finished processing")
with open('./data/GTZAN/Data/valid_filtered.txt', 'r') as f:
  val_list = f.readlines()
  extra_list = []
  for line in val_list:
    newline = line.strip()
    parts = line.split('/')
    parts[0] += '2'
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
    parts[0].replace('2', '3')
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
  val_list += extra_list
print("val_list finished processing")
with open('./data/GTZAN/Data/test_filtered.txt', 'r') as f:
  test_list = f.readlines()
  
  extra_list = []
  for line in test_list:
    newline = line.strip()
    parts = line.split('/')
    parts[0] += '2'
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
    parts[0].replace('2', '3')
    newline = ('/'.join(parts) + '\n')
    extra_list.append(newline)
  test_list += extra_list
print("Test_list finished processing")

label_set = np.loadtxt('./data/GTZAN/Data/GTZAN_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# generate  json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')
    base_path = './data/GTZAN/Data/'
    for split in ['test_filtered', 'valid_filtered', 'train_filtered']:
        wav_list = []
        with open(base_path+split+'.txt', 'r') as f:
            filelist = f.readlines()
        for file in filelist:
            cur_label = label_map[file.split('/')[0]]
            cur_path = os.path.abspath(os.getcwd()) + '/data/GTZAN/Data/' + file.strip()
            cur_dict = {"wav": cur_path, "labels": '/m/gtzan'+cur_label.zfill(2)}
            wav_list.append(cur_dict)
        if split == 'train_filtered':
            with open('./data/datafiles/GTZAN_train_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'test_filtered':
            with open('./data/datafiles/GTZAN_eval_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'valid_filtered':
            with open('./data/datafiles/GTZAN_valid_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

    print('GTZAN dataset processing finished.')


