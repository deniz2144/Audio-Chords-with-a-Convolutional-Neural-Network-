import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
import cv2
import os 
from glob import glob
import librosa 

y,sr=librosa.load("C:\\Users\\deniz\\Desktop\\musical chord\\Audio_Files\\Major\\Major_6.wav")

print("10 first values of the audio")
print(y[:10])

print(y.shape)

D=librosa.stft(y)
S_db=librosa.amplitude_to_db(np.abs(D), ref=np.max)

audioAsImage=(S_db * 255).astype(np.uint8)

cv2.imwrite('C://desktop//Major_wav.png',audioAsImage)

Path_for_Major_Spectogram = "C:\\Users\\deniz\\Desktop\\musical chord\\New\\Major"
# Majör klasör için yeni bir klasör oluşturma
isExist = os.path.exists(Path_for_Major_Spectogram)
if not isExist:
    os.makedirs(Path_for_Major_Spectogram)
    print("Yeni Majör klasörü oluşturuldu ")
    
    
# Majör ses dosyalarını işleme
MajorAudioFiles = glob("C:\\Users\\deniz\\Desktop\\musical chord\\Audio_Files\\Major\\*.wav")

for file in MajorAudioFiles:
    # Ses dosyasını yükleme
    y, sr = librosa.load(file)

    # Kısa zamanlı Fourier dönüşümü (STFT) hesaplama
    D = librosa.stft(y)

    # Desibel ölçeğine dönüştürme
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Görüntüye dönüştürme
    ImageAudio = (S_db * 255).astype(np.uint8)

    # Dosya adını çıkarma
    idx = file.rfind("\\")
    filename = file[idx + 1:]

    # Görüntüyü kaydetme
    cv2.imwrite(Path_for_Major_Spectogram + "//" + filename + ".png", ImageAudio)
    print(filename)
    
Path_for_Minor_Spectogram = "C:\\Users\\deniz\\Desktop\\musical chord\\New\\Minor"
# Majör klasör için yeni bir klasör oluşturma
isExist = os.path.exists(Path_for_Minor_Spectogram)
if not isExist:
    os.makedirs(Path_for_Minor_Spectogram)
    print("Yeni Minor klasörü oluşturuldu ")
       
       
# minor
MinorAudioFiles = glob("C:\\Users\\deniz\\Desktop\\musical chord\\Audio_Files\\Minor\\*.wav")

for file in MinorAudioFiles:
    # Ses dosyasını yükleme
    y, sr = librosa.load(file)

    # Kısa zamanlı Fourier dönüşümü (STFT) hesaplama
    D = librosa.stft(y)

    # Desibel ölçeğine dönüştürme
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Görüntüye dönüştürme
    ImageAudio = (S_db * 255).astype(np.uint8)

    # Dosya adını çıkarma
    idx = file.rfind("\\")
    filename = file[idx + 1:]

    # Görüntüyü kaydetme
    cv2.imwrite(Path_for_Minor_Spectogram + "//" + filename + ".png", ImageAudio)
    print(filename)