⚙️Predictive Maintenance using Acoustic Emissions

![image](https://github.com/user-attachments/assets/2107bf4e-0d00-480b-9971-34d0fb495673)

⚙️Predictive Maintenance using Acoustic Emissions

image.png

About the project
This project was carried out under the supervision of a doctoral student in mechanics, a teacher researcher in mechanics and by a student in the field of data sciences (myself) of the University of Technology of Compiègne (UTC) with the aim of studying the various existing tracks for predictive maintenance and their way of development.

About predictive maintenance
In order to retain customers, increase profits and lower production costs, it is essential for it is essential for a company to be competitive on the market and to optimize and to optimize its maintenance process. Having a reliable production line with automated maintenance allows to change parts at the right time to avoid breakdowns and possible malfunctions.

When we talk about quality management system, the ISO 1900 standard is essential. Planning maintenance allows to respect the requirements of this system of this system and to gain the confidence of its suppliers and customers. It is a a recognized guarantee of quality that allows industries to be more efficient in their their organization. By anticipating maintenance, you increase customer satisfaction because stock and production shortages are avoided. In addition, you increase your internal efficiency by reducing repair costs, by increasing the life span of the infrastructure and the working conditions.

On the surface, the maintenance categories all look interchangeable, but there are nuanced differences between reactive, preventive, proactive, and predictive maintenance.

Reactive: Reactive maintenance is exactly what it sounds like. Equipment goes down or malfunctions and needs immediate repair. Most safety and equipment teams understand that being overly reliant on a singularly reactive maintenance strategy is costly and potentially dangerous. How you respond to those issues is a bigger part of your planned maintenance strategies. Put another way, when you plan for problems ahead of time, the less likely you are to have many stack up and shut down production.
Preventive: Preventive maintenance includes a regular, scheduled program for each piece of equipment in your system. It’s usually scheduled at different intervals and can be labor intensive. Equipment manufacturers usually include a recommended preventive maintenance schedule. Preventive maintenance is almost always more cost-effective than reactive maintenance because it can prolong the life of parts and equipment. While preventive maintenance manages and reduces risk, data and measurement today can lead to even more cost-effective programs.
Predictive: Preventive and predictive maintenance sound interchangeable, but there is a nuanced difference. Preventive maintenance is executed at specified intervals, while predictive maintenance uses data and performance metrics from the equipment itself. By looking for algorithmic trends, your team and partners will get a better idea of what failures could be lurking, and what parts and issues can actually be left alone. Indicators like temperature, pressure, vibration, and other data points indicate issues that should be addressed immediately. Scheduled (or preventive) maintenance relies on a series of assumptions about equipment. Predictive maintenance could be more cost effective, without scheduled down-time and repair work that may in fact not be necessary.

Proactive: A proactive strategy combines predictive and preventive approaches by leveraging baseline performance numbers, monitoring equipment over time, and establishing a strategy to maintain equipment only when it’s needed. A proactive maintenance strategy goes beyond an established schedule. It addresses the common root issues of failure, to give you a holistic and comprehensive plan that addresses issues before they happen. Most organizations face big challenges when deciding how to allocate resources toward a maintenance program. Comparing and analyzing data from both new and aging equipment highlights likely failures. A proactive analysis and scheduled inspections put you in control of your maintenance in an intelligent way that also decreases cost over time. Working with a qualified safety expert further ensures that your equipment is satisfying safety requirements for your insurance carriers and approvals from various safety agencies.
Aim of the project
Here the goal is to design a predictive maintenance system by predicting the occurrence of defects in motors using the noise emitted by the motors to determine the type of problem. There are 3 types of defects:

![image](https://github.com/user-attachments/assets/c5ed3079-5320-49be-ad9b-3db166980f5d)

When the engine rotates, the sounds is different and so if we record the sound emitted by the engine, we can predict whether or not the engine is broken and what kind of problem there is.

During this project, I will study three different to answer the initial problem: one using CNN, an other one using feature extraction and a combinaison of both solutions.

About the data
The data consists in 4 audios of 12 seconds each of the different engines.

1. Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import librosa, IPython
import librosa.display as lplt
from scipy import io, misc
import scipy
import tensorflow as tf
import glob
from PIL import Image
import os
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.preprocessing.image import ImageDataGenerator
import random
import keras.backend as K
import keras
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

2. Load the data

Parameters
Every sample will be 0.2 seconds long and is recorded at a 10000Hz frequency.

duration_samples = 0.2 #seconds
size_max = 120000
frequence = 10000 #Hertz

Loading

file = "../input/engine-acoustic-emissions/dataset.mat"
dataset = scipy.io.loadmat(file)

df_normal = dataset["normal"].reshape(-1)[:size_max]
df_inner = dataset["inner"].reshape(-1)[:size_max]
df_roller = dataset["roller"].reshape(-1)[:size_max]
df_outer = dataset["outer"].reshape(-1)[:size_max]

data = [df_normal,df_inner,df_roller,df_outer]

def load_data(train=0.7):
    type_track = 0
    n_samples_each = int(size_max/frequence/duration_samples)
    audios_train = []
    audios_test = []
    number_train=int(n_samples_each*0.7)

    for track in data:
        for i in range(0,n_samples_each):
            t1 = int(i*frequence*duration_samples)
            t2 = int((i+1)*frequence*duration_samples)
            new = list(track)[t1:t2]
            if i<number_train:
                audios_train.append((type_track,new))
            else:
                audios_test.append((type_track,new))
        type_track = type_track+1
    np.random.seed(1)
    np.random.shuffle(audios_train)
    np.random.seed(1)
    np.random.shuffle(audios_test)
    return [i[1] for i in audios_train], [i[1] for i in audios_test], [i[0] for i in audios_train], [i[0] for i in audios_test]

audios_train, audios_test, label_train, label_test = load_data()

The audios
IPython.display.Audio(df_normal, rate=frequence)



