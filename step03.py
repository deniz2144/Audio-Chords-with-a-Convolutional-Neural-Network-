import cv2
import sounddevice as sd
import librosa
import tensorflow as tf
import numpy as np 

# Modelin yüklenmesi
best_model_file="C:\\Users\deniz\\Desktop\\musical chord\\Audio_Files\\Audio-Mijor-Minor.h5"
model = tf.keras.models.load_model('best_model.h5')
print(model.summary())

shape=(97,1025)
from tensorflow.keras.utils import img_to_array

# Ses dosyasının okunması ve ön işleme
def prepare_audio(path_for_audio):
    # Ses dosyasını oku
    y, sr = librosa.load(path_for_audio)
    
    D=librosa.stft(y)

    # Ses dalgasını dB'ye dönüştür
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max())
    
    ImageAudio=(S_db * 255).astype(np.uint8)
    
    resizedImage=cv2.resize(ImageAudio,shape,interpolation=cv2.INTER_AREA)
    
    imgResult =img_to_array(resizedImage)
    imgResult=np.expand_dims(imgResult,axis=0)
    imgResult=imgResult /255.
    return imgResult


testAudio="C:\\Users\\deniz\\Desktop\\musical chord\\Audio_Files\\Major\\Major_11.wav"
audio_file ,sr=librosa.load(testAudio)
sd.play(audio_file,sr)
sd.wait()

imageForModel=prepareAudio(testAudio)
resultArray= model.predict(imageForModel,verbose=1)

print(resultArray)

answer =resultArray[0][0]

if answer < 0.5:
    print("Major chord")
else :
    print("Minor chord")