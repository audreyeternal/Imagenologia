# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:48:23 2020

@author: lara_
"""
#%%
from PIL import Image
from numpy import matlib
from skimage.transform import rotate
from skimage import data
from scipy import signal
from scipy import fft
from scipy import ifft
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from drawnow import drawnow, figure

""""ACTUALIZACIÓN IMAGEN"""
def dibuja():
    plt.subplot(121), plt.imshow(RECONSTRUCTEDMATRIX,cmap='gray')
    plt.subplot(122), plt.imshow(DATA,cmap='gray')    

#Ima = data.camera()
Ima = np.array(Image.open("Phantom.png").convert('L'))
C = Ima[0:255,0:255];
a=np.zeros((128,511)); b=np.zeros((255,128))
C = np.concatenate((b,C,b),axis=1)  #Concatenar columnas
C = np.concatenate((a,C,a),axis=0)  #Concatenar filas
plt.figure(10),plt.imshow(C,cmap='gray')

C = np.double(C)/255
D = 300

maxiangle = np.arctan(256/((D-255)+1));
B = np.zeros((511,511));
j=0;
beta=[]

t=[]
range1 = np.arange(0,np.round(maxiangle/0.005)*0.005,0.005)
range2 = np.array(sorted(-1*range1))
ang =  np.arange(0,2*np.pi,2*np.pi/359)
angdeg = np.round((ang/np.pi)*180)
delta=np.concatenate((range2,range1[0:len(range1)]),axis=0)

fanbeam = np.zeros((len(angdeg),len(delta))) #j,i
print("Fanbeam generation")
for ang in angdeg[0:len(angdeg)-1]:
    print(ang)
    #t está mas abajo
    i = 0 #i=0
    A1 = rotate(C,ang,resize=False)
    #print (ang)
    #Computation of Beta
    beta.append(ang)
    #Computation of delta
    delta=np.concatenate((range2,range1[0:len(range1)]),axis=0)
    COL=np.zeros((len(delta),2,511)) #dim , fila, 
    t=np.zeros(len(delta))
    for rango in delta:
        s=0
        X=np.array(np.arange(-255,256,1))
        Y = np.round((D-X)*np.tan(rango))
        Y1 = -1*(Y+256)+512
        X1 = -1*(X+256)+512
        COL[i,:,:] = np.concatenate(([X1],[Y1]))   
        for u in range(1,512):
            if(X1[u-1]<=511 and Y1[u-1]<=511 and Y1[u-1]>0):
                B[np.int(X1[u-1])-1,np.int(Y1[u-1])-1]=255
                s=s+A1[np.int(X1[u-1])-1,np.int(Y1[u-1])-1]
        t[i]=s
        i=i+1
    fanbeam[j,:]=t
    t=0;
    j=j+1;

plt.figure(1), plt.imshow(B,cmap='gray')
plt.figure(2), plt.imshow(fanbeam,cmap='gist_rainbow')
fanbeamprojection=fanbeam
#%%
"""-----------------------Fan beam reconstruction--------------------------"""

#4.Multiply with the modified ramp filter
#5.Weighted back projection

DATA = np.zeros((511,511))
RECONSTRUCTEDMATRIX = np.zeros((511,511))
#Computation of weight matrix Lˆ(-2)
L_MATRIX= np.zeros((511,511))

for i in range(0,511):
    for j in range(0,511):
        k=-i+512-256
        l=-j+512-256
        L_MATRIX[j,i]=np.sqrt(k**2+(D-l)**2)

L_MATRIX=L_MATRIX**(-2)/np.max(np.max(L_MATRIX**(-2)))
DATA=0
plt.figure(2)

#Design of the weighted ramp filter
deltaterm = np.divide((delta**2),(np.sin(delta)**2))
deltaterm[278] = 1; deltaterm[279] = 1; deltaterm[280] = 1;
m=np.concatenate((np.array(np.linspace(0.1,1-1/558,557)),[0]),axis=0)
MODIMPULSERESPONSE = (1/2)*signal.firwin2(558,np.linspace(0,1,558),m,window='hamming')
MODIMPULSERESPONSE = np.multiply(MODIMPULSERESPONSE,deltaterm)
# tansformation of the weighted ramp filter after zero padding.
MODFILTER=fft(MODIMPULSERESPONSE,1117,norm="ortho")


fanbeamprojection_modified1=np.zeros((360,558))
fanbeamprojection_modified2=np.zeros((360,1117), dtype=complex)
fanbeamprojection_modified3=np.zeros((360,1117))



#%%
plt.figure(5)
DATA = np.zeros((511,511))
for k in range(0,359):
    RECONSTRUCTEDMATRIX = np.zeros((511,511))
    #Compuation of the Modified fanbeam projection
    fanbeamprojection_modified1[k] = np.multiply((D*np.cos(delta)),fanbeamprojection[k])
    #Fourier transformation of the modified fan beam projection after zero padding
    fanbeamprojection_modified2[k] = fft(fanbeamprojection[k],1117)
    #Multiplication with the fourier tansformation of the weighted ramp 
    #filter after zero padding 
    #Computation of inverese fourier transformation
    fanbeamprojection_modified3[k]= ifft(np.multiply(fanbeamprojection_modified2[k],MODFILTER))
    #Backprojection in the fanbeam structure
    l=0
    for rango in np.arange(-2*maxiangle,2*maxiangle,4*maxiangle/1116):
        X=np.arange(-255,256) #-255 a 255
        Y=np.round((D-X)*np.tan(rango))
        Y1=-1*(Y+256)+512
        X1=-1*(X+256)+512
        for u in range(0,510):
            if(X1[u]<=511 and Y1[u]<=511 and Y1[u]>0):
                RECONSTRUCTEDMATRIX[np.int(X1[u])-1,np.int(Y1[u])-1] = fanbeamprojection_modified3[k][l]
        l=l+1
    #Multiplication with the weight matrix
    RECONSTRUCTEDMATRIX = np.multiply(RECONSTRUCTEDMATRIX,L_MATRIX)
    #Obtaining the reconstructed matrix for beta(k)
    RECONSTRUCTEDMATRIX = rotate(RECONSTRUCTEDMATRIX,-np.array(beta[k]))
    plt.show()
    time.sleep(0.5)
    DATA=DATA+RECONSTRUCTEDMATRIX
    drawnow(dibuja)
    print(k)
    
plt.figure(3)
plt.imshow(DATA)
