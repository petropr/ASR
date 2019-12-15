import librosa
import scipy
import sklearn
import matplotlib.pyplot as plt

from librosa import display

from scipy.io.wavfile import write

import numpy as np



scaler=sklearn.preprocessing.StandardScaler()

def wav2mfcc(file_path):
    n_mfcc=20
    #sample_directory = 'audio/'
    #file_path = sample_directory + file_name
    x_one, sr_one = librosa.load(file_path)
    audio_mfcc= librosa.feature.mfcc(x_one, sr=sr_one, n_mfcc=n_mfcc).T
    mfcc_scaled = scaler.fit_transform(audio_mfcc)
    mfcc_scaled.mean(axis=0)
    mfcc_scaled.std(axis=0)

    return mfcc_scaled, x_one


def silence_audio(file_name):
    sample_directory = 'audio/'
    file_path = sample_directory + file_name
    x_one, sr_one = librosa.load(file_path)
    thres=0.08
    silenced_list=[]
    flag=False
    skip_counter=0
    for pos,value in enumerate(x_one):
        if flag==True:
            skip_counter = skip_counter-1
            if skip_counter==0:
                flag=False
            continue
        if abs(value)<=thres:
            count=0
            for y in x_one[pos + 1 : pos + 5001]:
                count=count+1
                temp=count
                if abs(y)>thres:
                    break
            if count==5000:
                for q in range(0,5001):
                    silenced_list.append(0)
                flag=True
                skip_counter=count
            else:
                if pos+count==len(x_one)-1:
                    for i in range(0,count+1):
                        silenced_list.append(0)
                else:
                    for w in x_one[pos:temp+pos+1]:
                        silenced_list.append(w)
                flag=True
                skip_counter=count
        else:
            silenced_list.append(value)
    write('silenced_'+file_name , sr_one, np.array(silenced_list))

    return silenced_list


def split_audio(list):

    words=[]
    starend=[]

    # Thewroume oti to wav panta ksekinaei me paush(oxi logia)
    f=False
    for pos,value in enumerate(list):
        if f==False:
            if value!=0:
                starend.append(pos)
                f=True
        else:
            if value==0:
                starend.append(pos)
                f=False

    for z,val in enumerate(starend):
        if z==len(starend)-1:
            break
        if z%2!=0:
            continue
        words.append(list[val:starend[z+1]])
    for i,value in enumerate(words):
        #print (i)
        sr_one=22050
        write('word_'+ str(i) + '.wav', sr_one, np.array(words[i]))

    return words

def print_words(t_split,model):
    printed_words = []
    for i,value in enumerate(test_split):
        scaled_test,x_0 = wav2mfcc('word_'+str(i) +'.wav')
        predicted_labels = model.predict(scaled_test)

        unique_labels, unique_counts = np.unique(predicted_labels, return_counts=True)
        score = unique_labels[np.argmax(unique_counts)]
        if score == 0.0:
            printed_words.append('kalimera')
        elif score == 1.0:
            printed_words.append('autokinito')
        elif score == 2.0:
            printed_words.append('kefali')
        elif score == 3.0:
            printed_words.append('podosfairo')

    print (' '.join(printed_words))





#-----------------------------Make Dictionary--------------------------------------------

first_word,x_1 = wav2mfcc('audio/kalimera.wav')
sec_word,x_2 = wav2mfcc('audio/autokinito.wav')
third_word,x_3 = wav2mfcc('audio/kefali.wav')
fourth_word,x_4 = wav2mfcc('audio/podosfairo.wav')

print('-----------Dictionary-----------')
print('kalimera, autokinito, kefali, podosfairo')
#-----------------------------TRAIN Classifier-------------------------------------------
features = np.vstack((first_word,sec_word ,third_word,fourth_word))
features.shape
labels = np.concatenate((np.zeros(len(first_word)), np.ones(len(sec_word)), np.full(len(third_word),2), np.full(len(fourth_word),3)))
labels.shape
model = sklearn.svm.SVC()
model.fit(features, labels)
print('-----------Prediction-----------')
for x in range(0,4):
    print('For test'+str(x)+'.wav :')
    test_silenced = silence_audio('test'+str(x)+'.wav')
    test_split = split_audio(test_silenced)
    print_words(test_split,model)
