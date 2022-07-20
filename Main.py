
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:21:03 2022

@author: holger
"""

import torch 


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cargar_datos import cargar_datos_x1,cargar_datos_x2, cargar_datos_y


from scipy.fftpack import fft,ifft
def hilbert_from_scratch(u,Nt,Nx):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)
    v=np.zeros((Nt,Nx),dtype = "complex_")
    for i in np.arange(0,Nx):
     N = len(u[:,i])
     # take forward Fourier transform
     U = fft(u[:,i])
     M = N - N//2 - 1
     # zero out negative frequency components
     U[N//2+1:] = [0] * M
     # double fft energy except @ DC0
     U[1:N//2] = 2 * U[1:N//2]
     # take inverse Fourier transform
     v[:,i] = ifft(U)
    return v

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["gray", "red"])

#%% Cargar datos X

G=cargar_datos_x1()
H=cargar_datos_x2()
(n,Nt,Nx)=G.shape
J=np.zeros((127,1500,334,2))
J[:,:,:,0]=H[:,:,:]
J[:,:,:,1]=G[:,:,:]
#%% Cargar datos Y

C=cargar_datos_y()

#%%Datos de entrada

EPOCHS=50
LR=0.0001

IMG_SIZET=320#OJO por que solo 320?
IMG_SIZEX=320
BATCH_SIZE=6

ENCODER='timm-efficientnet-b0'
WEIGHTS='imagenet'



dt=2e-3#time sampling
fs=1/dt
dx=6 #Separacion de receptores
t=np.arange(0,Nt*dt,dt); 
x=np.arange(0,Nx*dx,dx); 





#%%Crear df (puedo hacerlo con for)

df=pd.DataFrame()
df['gather']=None
df['atributo_rm']=None
for i in np.arange(n):
    df.loc[i]=[J[i,:,:,:],C[i,:,:]]
    
    


#%% Ver imagen del df
'''
row=df.iloc[19]
image_path=row.gather
mask_path=row.atributo_rm

#YA ESTA LISTO PARA GRAFICAR


plt.figure(1)
plt.subplot(131)
plt.imshow(image_path[:,:,0],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Horizontal')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')


plt.subplot(132)
plt.imshow(image_path[:,:,1],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Vertical')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')


plt.subplot(133)
plt.imshow(image_path[:,:,1],aspect='auto',vmin=-1e-3,vmax=1e-3,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.imshow(mask_path[:,:],aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cmap,alpha = 0.5)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributo de dispersión')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')


plt.show()
'''


#%%IMAGENES
row=df.iloc[25]
image_path=row.gather
mask_path=row.atributo_rm

#YA ESTA LISTO PARA GRAFICAR


plt.figure(1)
plt.subplot(131)
plt.imshow(image_path[:,:,0],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Horizontal')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(132)
plt.imshow(image_path[:,:,1],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Vertical')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(133)
plt.imshow(mask_path[:,:],aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.binary)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributo de dispersión')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.show()


#%% Definir datos de entrenamiento y los datos de validar

train_df, valid_df =train_test_split(df, test_size=0.01, random_state=42)



#%%
import albumentations as A

def get_train_augs():
  return A.Compose([
          A.Resize(IMG_SIZET, IMG_SIZEX),  
  ])

  
def get_valid_augs():
  return A.Compose([
          A.Resize(IMG_SIZET, IMG_SIZEX),  
  ])
#%% Creando el data_set
from torch.utils.data import Dataset



class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations):

    self.df=df
    self.augmentations=augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    row=self.df.iloc[idx]
    image=row.gather
    mask=row.atributo_rm


    mask=np.expand_dims(mask,axis=-1)

    if self.augmentations:
      data=self.augmentations(image=image,mask=mask)
      image=data['image']
      mask=data['mask']

    #Cambio de posicion
    image=np.transpose(image,(2,0,1))
    mask=np.transpose(mask,(2,0,1))   



    image=torch.Tensor(image)
    mask=torch.round(torch.Tensor(mask))

    return image,mask

#%% obtener datos automaticamente

trainset=SegmentationDataset(train_df, get_train_augs())
validset=SegmentationDataset(valid_df, get_valid_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")




#%%ver hasta donde va idx
idx=12

image,mask=trainset[idx]

print(image.shape)
print(mask.shape)
#%%


mask=mask[0,:,:]
plt.figure(2)
plt.subplot(131)
plt.imshow(image[0,:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Vertical gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(132)
plt.imshow(image[1,:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Vertical gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(133)
plt.imshow(mask[:,:],aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.binary)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributte_RM')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.show()


#%%
from torch.utils.data import DataLoader

trainloader=DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader=DataLoader(validset, batch_size=BATCH_SIZE)


print(f'numero total de batches en el cargador de entrenamiento: {len(trainloader)}')
print(f'numero total de batches en el cargador de validacion: {len(validloader)}')

for image, mask in trainloader:
  break

print(f'Un batches de la forma de la imagen: {image.shape}')
print(f'Un batches de la forma de la mascara: {mask.shape}')

#%%modelo

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss


class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel,self).__init__()

    self.arc=smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=WEIGHTS,
        in_channels=2,
        classes=1,
        activation=None
    )
  def forward(self,images, masks=None):
    logits=self.arc(images)

    if masks!=None:
      loss1=DiceLoss(mode='binary')(logits,masks)
      loss2=nn.BCEWithLogitsLoss()(logits,masks)
      return logits,loss1+loss2
      
    return logits


model=SegmentationModel()


#%% funciones de entrenamiento y validacion
def train_fn(data_loader,model,optimizer):
  model.train()
  total_loss=0.0

  for images,masks in tqdm(data_loader):
    

    optimizer.zero_grad()
    logits,loss=model(images,masks)
    loss.backward()
    optimizer.step()


    total_loss+=loss.item()

  return total_loss/len(data_loader)

def eval_fn(data_loader,model):
  model.eval()
  total_loss=0.0
  with torch.no_grad(): 
   for images,masks in tqdm(data_loader):


      logits,loss=model(images,masks)



      total_loss+=loss.item()

  return total_loss/len(data_loader)

#%%Optimizador

optimizer=torch.optim.Adam(model.parameters() , lr=LR)

#%%

best_valid_loss=np.Inf

for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss=eval_fn(validloader,model)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(),'best_model.pt')
    print('SAVED model')
    best_valid_loss=valid_loss
    
  print(f'Epoch:{i+1} Train_loss: {train_loss} Valid_loss: {valid_loss}')
  
  
#%%Predictor
idx=1

model.load_state_dict(torch.load('best_model.pt'))

image,mask=validset[idx]

logits_mask=model(image.unsqueeze(0))
pred_mask=torch.sigmoid(logits_mask)
pred_mask=(pred_mask>0.5)*1.0


mask=np.array(mask[0,:,:])
pred_mask=np.array(pred_mask[0,0,:,:])



plt.figure(3)
plt.subplot(141)
plt.imshow(image[0,:,:],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Horizontal')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal',ticks=[-1e-3, 0, 1e-3])

plt.subplot(142)
plt.imshow(image[1,:,:],aspect='auto',vmin=-1e-3,vmax=1e-3, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Vertical')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal',ticks=[-1e-3, 0, 1e-3])

plt.subplot(143)
plt.imshow(mask[:,:],aspect='auto', extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.binary)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributo de \n Dispersión')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(144)
plt.imshow(pred_mask[:,:],aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.binary)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributo de \n Dispersión (predicho)')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=.9, 
                    wspace=0.5, 
                    hspace=0.95)
plt.show()


k=0
for i in np.arange(IMG_SIZET):
    for j in np.arange(IMG_SIZEX):
        if pred_mask[i,j]==1:
            k+=1
SN=100*k/(IMG_SIZET*IMG_SIZEX)
print(f'SN%: {SN}%')


#%% sobre datos reales!
Horizontal=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelo_tenerife\aleatorio\dataX.npy')
Vertical=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelo_tenerife\aleatorio\dataZ.npy')

#Solucion a geofonos dañado
from funciones import damaged_geophone
Vertical=damaged_geophone(Vertical)
Horizontal=damaged_geophone(Horizontal)


(Nt,Nx)=Vertical.shape
dt=2e-3#time sampling
fs=1/dt
dx=10
t=np.arange(0,Nt*dt,dt); 
x=np.arange(0,Nx*dx,dx); 


#Correcion de ganancia de datos(La amplitud decae muy rapido)
Vertical1=Vertical
Horizontal1=Horizontal
for i in np.arange(Nx):
    for j in np.arange(Nt):
        Vertical1[j,i]=Vertical[j,i]*(t[j]*t[j])
        Horizontal1[j,i]=Horizontal[j,i]*(t[j]*t[j])
Vertical=Vertical1
Horizontal=Horizontal1

def normal(matrix):
    norm = np.linalg.norm(matrix[:,:])
    matrix[:,:] = matrix[:,:]/norm  # normalized matrix    
   # for k in np.arange(0,n):
    #  for i in np.arange(0,Nt):
     #   for j in np.arange(0,Nx):
      #      if matrix[i,j,k]==0:
       #         matrix[i,j,k]=np.nan
    return matrix


from funciones import arreglo_mascara

H=np.zeros(Vertical.shape)
V=np.zeros(Vertical.shape)

for i in np.arange(0,10):
    ans1=np.concatenate((np.arange(i,Horizontal.shape[0]-1),np.arange(Horizontal.shape[0]-(i+1),Horizontal.shape[0])),axis=0)
    ans2=np.concatenate((np.arange(i,Horizontal.shape[1]-1),np.arange(Horizontal.shape[1]-1-i,Horizontal.shape[1])),axis=0)
    H=abs(Horizontal[ans1[:,None],ans2])+H
    V=abs(Vertical[ans1[:,None],ans2])+V
    
mascara1=((H+V)>(1e-1)*np.mean(H+V))# es por defecto 1e-3
#Buscar funcion que haga la misma transformada Hilbert
#V=np.sqrt(Vertical*Vertical+np.imag(signal.hilbert2(Vertical))*np.imag(signal.hilbert2(Vertical)))
V=np.abs(hilbert_from_scratch(Vertical,Nt,Nx))
#H=np.sqrt(Horizontal*Horizontal+np.imag(signal.hilbert2(Horizontal))*np.imag(signal.hilbert2(Horizontal)))
H=np.abs(hilbert_from_scratch(Horizontal,Nt,Nx))
mascara2=((H*H+V*V)>(1e-6)*np.mean(H*H+V*V))#por defecto 1e-6
mascara=mascara1*mascara2



#Para arreglar la mascara del dato real de tenerife
for i in np.arange(884,Nt):
    mascara[i,:]=True
mascara=arreglo_mascara(mascara,Nt,Nx)



mascara_1D=np.transpose(mascara).reshape(-1)
id_mascara=np.where(mascara_1D==True)
(idy,idx)=np.where(np.transpose(mascara)==True)
ind=np.where(mascara_1D==True)
ind=np.asarray(ind)


Vertical=normal(Vertical)
Horizontal=normal(Horizontal)
for i in np.arange(Nt):
    for j in np.arange(Nx):
        if mascara[i,j]==False:
            Vertical[i,j]=0
            Horizontal[i,j]=0


plt.figure(4)
plt.imshow(Vertical[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('(a)')
plt.ylabel(r'Tiempo [s]')
plt.xlabel(r'Distancia [m]')
plt.colorbar(orientation='vertical')
plt.show()


(Nt,Nx)=Vertical.shape
J=np.zeros((Nt,Nx,2))
J[:,:,0]=Vertical[:,:]
J[:,:,1]=Horizontal[:,:]


df=pd.DataFrame()
df['gather']=None
df['atributo_rm']=None
for i in np.arange(1):
    df.loc[i]=[J,Vertical]
    

validf=SegmentationDataset(df, get_valid_augs())
image,mask=validf[0]


logits_mask=model(image.unsqueeze(0))
pred_mask=torch.sigmoid(logits_mask)
pred_mask=(pred_mask>0.5)*1.0


mask=np.array(mask[0,:,:])
pred_mask=np.array(pred_mask[0,0,:,:])



plt.figure(5)
plt.subplot(131)
plt.imshow(image[0,:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Horizontal')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(132)
plt.imshow(image[1,:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Registro Vertical')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(133)
plt.imshow(pred_mask[:,:],aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.binary)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Atributo de \n dispersión (Predicho)')
plt.ylabel(r'Tiempo (s)')
plt.xlabel(r'Distancia (m)')
plt.colorbar(orientation='horizontal')

plt.show()

k=0
for i in np.arange(IMG_SIZET):
    for j in np.arange(IMG_SIZEX):
        if pred_mask[i,j]==1:
            k+=1
SN=100*k/(IMG_SIZET*IMG_SIZEX)
print(f'SN%: {SN}%')
#%% Para ver graficas

V1=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=10 y e=0.4\gather_z.npy')
V2=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(der)\gather_z.npy')
V3=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (izq)\gather_z.npy')
V4=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=10 y e=0.5\gather_z.npy')

(Nt,Nx)=V1.shape
Dx=2000; Dz=100;
dt=2e-3#time sampling
fs=1/dt
dx=6 #Separacion de receptores
t=np.arange(0,Nt*dt,dt); 
x=np.arange(0,Nx*dx,dx); 

plt.figure(3)
plt.subplot(221)
plt.imshow(V1[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('(a)')
plt.ylabel(r'Tiempo [s]')
plt.xlabel(r'Distancia [m]')
plt.colorbar(orientation='vertical')

plt.subplot(222)
plt.imshow(V2[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('(b)')
plt.ylabel(r'Tiempo [s]')
plt.xlabel(r'Distancia [m]')
plt.colorbar(orientation='vertical')

plt.subplot(223)
plt.imshow(V3[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('(c)')
plt.ylabel(r'Tiempo [s]')
plt.xlabel(r'Distancia [m]')
plt.colorbar(orientation='vertical')

plt.subplot(224)
plt.imshow(V4[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('(d)')
plt.ylabel(r'Tiempo [s]')
plt.xlabel(r'Distancia [m]')
plt.colorbar(orientation='vertical')

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=.9, 
                    wspace=0.2, 
                    hspace=0.35)

plt.show()
