# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:46:00 2022

@author: HOLGER
"""
import numpy as np
import matplotlib.pyplot as plt


#plt.rc('text', usetex=False)
def eliminar_mitad(matrix):
    (Nt,Nx,n)=(matrix.shape[0],matrix.shape[1],matrix.shape[2])
    matrix_new=np.zeros((Nt,int(Nx/2),19))
    for k in np.arange(n):
        A=matrix[:,:,k]
        matrix_new[:,:,k]=np.delete(A,np.arange(int(Nx/2)),axis=1)

    return matrix_new


def atributo_rm(matrix):
    (Nt,Nx)=(matrix.shape[0],matrix.shape[1])
    A=np.delete(matrix[:,:,0],np.arange(334,668),axis=1)
    B=np.delete(matrix[:,:,0],np.arange(0,334),axis=1)
    D=np.abs(B-A)
    for i in np.arange(0,Nt):
       for j in np.arange(0,334):
            if D[i,j]==0 or np.isnan(D[i,j]):
                D[i,j]=0
            else:
                D[i,j]=1
    return D


def flatten(matrix):
    (Nt,Nx,n)=(matrix.shape[0],matrix.shape[1],matrix.shape[2])
    matrix_new=np.zeros((n,Nt*Nx))
    for k in np.arange(n):
        matrix_new[k,:]=matrix[:,:,k].flatten()
    return matrix_new


def normalize_2d(matrix):
    (n,Nt,Nx)=(matrix.shape[0],matrix.shape[1],matrix.shape[2])
    for i in np.arange(n):
        norm = np.linalg.norm(matrix[i,:,:])
        matrix[i,:,:] = matrix[i,:,:]/norm  # normalized matrix
       
   # for k in np.arange(0,n):
    #  for i in np.arange(0,Nt):
     #   for j in np.arange(0,Nx):
      #      if matrix[i,j,k]==0:
       #         matrix[i,j,k]=np.nan
    return matrix

def damaged_geophone(U):
    k=0
    for i in np.arange(U.shape[1]):
        if np.unique(U[:,i]).size==1:
            k=k+1
    temp=np.zeros(k)
    k=0
    for i in np.arange(U.shape[1]):
        if np.unique(U[:,i]).size==1:
            temp[k]=np.int(i)
            k=k+1
    ans=np.delete(U,temp.astype(int),axis=1) 
    return ans

def graph(F,x,z):
    plt.figure(1)
    plt.ion()
    plt.imshow(F,aspect='auto', extent=[x[0],x[x.shape[0]-1],z[z.shape[0]-1],z[0]],origin='upper')
    plt.ylabel(r'Distance (m)')
    plt.xlabel(r'Distance (m)')
    plt.colorbar(label='Data [m/s]')
    plt.ioff()
    plt.show()

    return


def interpol(F):
    (Nz,Nx)=F.shape
    #Vp
    a=np.linspace(F[0],F[1],4)
    for i in np.arange(1,Nz-1):
      b=np.linspace(F[i],F[i+1],4)
      p=np.concatenate((a,b),axis=0)
      a=p
    F=np.transpose(a)
    a=np.linspace(F[0],F[1],4)
    for i in np.arange(1,Nx-1):
      b=np.linspace(F[i],F[i+1],4)
      p=np.concatenate((a,b),axis=0)
      a=p
    F=np.transpose(a)  
    return(F)


def delete_outliers(att):
    
    Q1 = np.percentile(att[:,:], 25, interpolation = 'midpoint')
    Q3 = np.percentile(att[:,:], 75, interpolation = 'midpoint')
    IQR=Q3-Q1
    (m,n)=np.shape(att[:,:])
    minimum=Q1-1.5*IQR
    maximum=Q3+1.5*IQR
    
    for i in np.arange(m):
        for j in np.arange(n):
            if att[i,j]<minimum:
                att[i,j]=minimum
                
                
            elif att[i,j]>maximum:
                att[i,j]=maximum
                

                
    
    return(att)

def arreglo_mascara(mascara,Nt,Nx):  
    temp1=864   
    temp2=0
    k=401
    for i in np.arange(0,k):
            ans=np.asarray(np.where(mascara[:,i]==True))
            primero=ans[0,0]
            if primero<=temp1:
                for j in np.arange(primero,Nt):
                    mascara[j,i]=True
                temp1=primero
            else:
                primero=temp1
                for j in np.arange(primero,Nt):
                    mascara[j,i]=True
    for i in np.arange(k,Nx):
            ans=np.asarray(np.where(mascara[:,i]==True))
            primero=ans[0,0]
            if primero>=temp2:
                for j in np.arange(primero,Nt):
                    mascara[j,i]=True
                temp2=primero
            else:
                primero=temp2
                for j in np.arange(primero,Nt):
                    mascara[j,i]=True        
        
    
    return(mascara)