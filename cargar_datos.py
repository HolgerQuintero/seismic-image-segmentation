# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:09:43 2022

@author: holger
"""

import numpy as np
from funciones import  atributo_rm, normalize_2d

#plt.rc('text', usetex=False)




def cargar_datos_x1():
    G1=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Modelo_base\gather_z.npy')
    G2=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=10 y e=0.4\gather_z.npy')
    G3=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=0.4\gather_z.npy')
    G4=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=2\gather_z.npy')
    G5=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=5 y e=0.6\gather_z.npy')
    G6=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=125 y e=0.2\gather_z.npy')
    G7=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=10 y e=0.6\gather_z.npy')
    G8=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=50 y e=1\gather_z.npy')
    G9=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-medio\a=15 y e=0.4\gather_z.npy')
    G10=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=5 y e=0.1\gather_z.npy')
    G11=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=30 y e=0.3\gather_z.npy')
    
    
    G12=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\modelo base\gather_z.npy')
    G13=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=10 y e=0.4\gather_z.npy')
    G14=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=50 y e=0.8\gather_z.npy')
    G15=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=1 y e=0.3\gather_z.npy')
    G16=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=30 y e=0.3\gather_z.npy')
    G17=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (cen)\gather_z.npy')
    G18=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (der)\gather_z.npy')
    G19=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (izq)\gather_z.npy')
    G20=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\abajo-medio\a=25 y e=0.2\gather_z.npy')



    G21=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\modelo_base\gather_z.npy')
    G22=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\centrado-medio\a=50 y e=1.2\gather_z.npy')
    G23=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (der)\gather_z.npy')
    G24=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (izq)\gather_z.npy')


    G25=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\modelo_base\gather_z.npy')
    G26=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\centrado-medio\a=30 y e=0.7\gather_z.npy')
    G27=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (der)\gather_z.npy')
    G28=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (izq)\gather_z.npy')
    G29=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba_completo\gather_z.npy')
    G30=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(der)\gather_z.npy')
    G31=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(izq)\gather_z.npy')
    G32=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-completo\a=10 y e=0.7\gather_z.npy')


    G33=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\modelo-base\gather_z.npy')
    G34=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(cen)\gather_z.npy')
    G35=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(der)\gather_z.npy')
    G36=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(izq)\gather_z.npy')
    G37=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(cen)\gather_z.npy')
    G38=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(der)\gather_z.npy')
    G39=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(izq)\gather_z.npy')
    G40=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=5 y e=0.7\gather_z.npy')
    G41=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=30 y e=1\gather_z.npy')
    G42=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\abajo-completo\a=15 y e=0.8\gather_z.npy')


    G43=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\modelo-base\gather_z.npy')
    G44=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(der)\gather_z.npy')
    G45=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(izq)\gather_z.npy')
    G46=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-completo\a=10 y e=0.7\gather_z.npy')
    G47=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\arriba-completo\gather_z.npy')
    G48=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\abajo-completo\a=10 y e=0.7\gather_z.npy')


    G49=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\Modelo_base\gather_z.npy')
    G50=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(cen)\gather_z.npy')
    G51=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(der)\gather_z.npy')
    G52=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(izq)\gather_z.npy')
    G53=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-completo\a=10 y e=0.7\gather_z.npy')
    G54=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=5 y e=0.3\gather_z.npy')
    G55=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=10 y e=0.7\gather_z.npy')
    G56=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=30 y e=0.3\gather_z.npy')

    G57=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\Modelo_base\gather_z.npy')
    G58=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(cen)\gather_z.npy')
    G59=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(der)\gather_z.npy')
    G60=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(izq)\gather_z.npy')
    G61=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=1 y e=0.3\gather_z.npy')
    G62=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=50 y e=0.3\gather_z.npy')
    G63=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=0.3\gather_z.npy')
    G64=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=1.4\gather_z.npy')
    G65=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=3\gather_z.npy')
    G66=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=0.3\gather_z.npy')
    G67=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=1.4\gather_z.npy')
    G68=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=3\gather_z.npy')
    G69=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=1 y e=0.3\gather_z.npy')
    G70=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=50 y e=0.3\gather_z.npy')

    G71=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\modelo base\gather_z.npy')
    G72=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(cen)\gather_z.npy')
    G73=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(der)\gather_z.npy')
    G74=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(izq)\gather_z.npy')
    G75=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=10 y e=0.6\gather_z.npy')
    G76=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=50 y e=0.6\gather_z.npy')
    G77=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(der)\gather_z.npy')
    G78=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(izq)\gather_z.npy')
    G79=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=1(der)\gather_z.npy')
    G80=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-completo\a=10 y e=0.3\gather_z.npy')
    G81=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=10 y e=0.6\gather_z.npy')
    G82=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=50 y e=0.6\gather_z.npy')


    G83=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\modelo base\gather_z.npy')
    G84=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(cen)\gather_z.npy')
    G85=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(der)\gather_z.npy')
    G86=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(izq)\gather_z.npy')
    G87=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-der)\gather_z.npy')
    G88=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-izq)\gather_z.npy')
    G89=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=2 y e=0.7\gather_z.npy')
    G90=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=25 y e=1\gather_z.npy')
    G91=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(der)\gather_z.npy')
    G92=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(izq)\gather_z.npy')
    G93=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(der)\gather_z.npy')
    G94=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(izq)\gather_z.npy')
    G95=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=0.5\gather_z.npy')
    G96=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=1.6\gather_z.npy')
    G97=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=0.5\gather_z.npy')
    G98=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=1.6\gather_z.npy')
    G99=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=10 y e=0.5\gather_z.npy')
    G100=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=100 y e=0.5\gather_z.npy')



    G101=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\base\bgather_z.npy')
    G102=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\a\gather_z.npy')
    G103=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\b\gather_z.npy')
    G104=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\c\gather_z.npy')
    G105=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\d\gather_z.npy')
    G106=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\e\gather_z.npy')
    G107=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\f\gather_z.npy')
    G108=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\g\gather_z.npy')
    G109=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\h\gather_z.npy')
    G110=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\a\gather_z.npy')
    G111=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\b\gather_z.npy')    
    G112=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\c\gather_z.npy')
    G113=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\d\gather_z.npy')
    G114=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\e\gather_z.npy')
    G115=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\f\gather_z.npy')
    G116=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\g\gather_z.npy')
    G117=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\h\gather_z.npy')
    G118=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\i\gather_z.npy')
    G119=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\a\gather_z.npy')
    G120=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\b\gather_z.npy')
    G121=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\c\gather_z.npy')    
    G122=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\d\gather_z.npy')
    G123=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\e\gather_z.npy')
    G124=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\f\gather_z.npy')
    G125=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\g\gather_z.npy')
    G126=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\h\gather_z.npy')
    G127=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\i\gather_z.npy')


    (Nt,Nx)=G1.shape
    G=np.zeros((127,Nt,Nx))

        
    G[0,:,:]=G1
    G[1,:,:]=G2
    G[2,:,:]=G3
    G[3,:,:]=G4
    G[4,:,:]=G5
    G[5,:,:]=G6
    G[6,:,:]=G7
    G[7,:,:]=G8
    G[8,:,:]=G9
    G[9,:,:]=G10
    G[10,:,:]=G11
    
    G[11,:,:]=G12
    G[12,:,:]=G13
    G[13,:,:]=G14
    G[14,:,:]=G15
    G[15,:,:]=G16
    G[16,:,:]=G17
    G[17,:,:]=G18
    G[18,:,:]=G19
    G[19,:,:]=G20
    
    G[20,:,:]=G21
    G[21,:,:]=G22
    G[22,:,:]=G23
    G[23,:,:]=G24
    
    G[24,:,:]=G25
    G[25,:,:]=G26
    G[26,:,:]=G27
    G[27,:,:]=G28
    G[28,:,:]=G29
    G[29,:,:]=G30
    G[30,:,:]=G31
    G[31,:,:]=G32
    
    G[32,:,:]=G33
    G[33,:,:]=G34
    G[34,:,:]=G35
    G[35,:,:]=G36
    G[36,:,:]=G37
    G[37,:,:]=G38
    G[38,:,:]=G39
    G[39,:,:]=G40
    G[40,:,:]=G41
    G[41,:,:]=G42
    
    G[42,:,:]=G43
    G[43,:,:]=G44
    G[44,:,:]=G45
    G[45,:,:]=G46
    G[46,:,:]=G47
    G[47,:,:]=G48

    G[48,:,:]=G49
    G[49,:,:]=G50
    G[50,:,:]=G51
    G[51,:,:]=G52
    G[52,:,:]=G53
    G[53,:,:]=G54
    G[54,:,:]=G55
    G[55,:,:]=G56
    
    G[56,:,:]=G57
    G[57,:,:]=G58
    G[58,:,:]=G59
    G[59,:,:]=G60
    G[60,:,:]=G61
    G[61,:,:]=G62
    G[62,:,:]=G63
    G[63,:,:]=G64
    G[64,:,:]=G65
    G[65,:,:]=G66
    G[66,:,:]=G67
    G[67,:,:]=G68
    G[68,:,:]=G69
    G[69,:,:]=G70
    
    G[70,:,:]=G71
    G[71,:,:]=G72
    G[72,:,:]=G73
    G[73,:,:]=G74
    G[74,:,:]=G75
    G[75,:,:]=G76
    G[76,:,:]=G77
    G[77,:,:]=G78
    G[78,:,:]=G79
    G[79,:,:]=G80
    G[80,:,:]=G81
    G[81,:,:]=G82
    
    G[82,:,:]=G83
    G[83,:,:]=G84
    G[84,:,:]=G85
    G[85,:,:]=G86
    G[86,:,:]=G87
    G[87,:,:]=G88
    G[88,:,:]=G89
    G[89,:,:]=G90 
    G[90,:,:]=G91
    G[91,:,:]=G92
    G[92,:,:]=G93
    G[93,:,:]=G94
    G[94,:,:]=G95
    G[95,:,:]=G96
    G[96,:,:]=G97
    G[97,:,:]=G98
    G[98,:,:]=G99
    G[99,:,:]=G100
    
    G[100,:,:]=G101
    G[101,:,:]=G102
    G[102,:,:]=G103
    G[103,:,:]=G104
    G[104,:,:]=G105
    G[105,:,:]=G106
    G[106,:,:]=G107
    G[107,:,:]=G108
    G[108,:,:]=G109
    G[109,:,:]=G110
    G[110,:,:]=G111
    G[111,:,:]=G112
    G[112,:,:]=G113
    G[113,:,:]=G114
    G[114,:,:]=G115
    G[115,:,:]=G116
    G[116,:,:]=G117
    G[117,:,:]=G118
    G[118,:,:]=G119
    G[119,:,:]=G120
    G[120,:,:]=G121
    G[121,:,:]=G122
    G[122,:,:]=G123
    G[123,:,:]=G124
    G[124,:,:]=G125
    G[125,:,:]=G126
    G[126,:,:]=G127

    
    G=normalize_2d(G)
    
    return G


def cargar_datos_x2():
    G1=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Modelo_base\gather_x.npy')
    G2=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=10 y e=0.4\gather_x.npy')
    G3=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=0.4\gather_x.npy')
    G4=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=2\gather_x.npy')
    G5=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=5 y e=0.6\gather_x.npy')
    G6=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=125 y e=0.2\gather_x.npy')
    G7=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=10 y e=0.6\gather_x.npy')
    G8=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=50 y e=1\gather_x.npy')
    G9=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-medio\a=15 y e=0.4\gather_x.npy')
    G10=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=5 y e=0.1\gather_x.npy')
    G11=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=30 y e=0.3\gather_x.npy')
    
    
    G12=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\modelo base\gather_x.npy')
    G13=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=10 y e=0.4\gather_x.npy')
    G14=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=50 y e=0.8\gather_x.npy')
    G15=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=1 y e=0.3\gather_x.npy')
    G16=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=30 y e=0.3\gather_x.npy')
    G17=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (cen)\gather_x.npy')
    G18=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (der)\gather_x.npy')
    G19=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (izq)\gather_x.npy')
    G20=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\abajo-medio\a=25 y e=0.2\gather_x.npy')



    G21=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\modelo_base\gather_x.npy')
    G22=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\centrado-medio\a=50 y e=1.2\gather_x.npy')
    G23=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (der)\gather_x.npy')
    G24=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (izq)\gather_x.npy')


    G25=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\modelo_base\gather_x.npy')
    G26=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\centrado-medio\a=30 y e=0.7\gather_x.npy')
    G27=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (der)\gather_x.npy')
    G28=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (izq)\gather_x.npy')
    G29=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba_completo\gather_x.npy')
    G30=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(der)\gather_x.npy')
    G31=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(izq)\gather_x.npy')
    G32=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-completo\a=10 y e=0.7\gather_x.npy')


    G33=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\modelo-base\gather_x.npy')
    G34=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(cen)\gather_x.npy')
    G35=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(der)\gather_x.npy')
    G36=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(izq)\gather_x.npy')
    G37=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(cen)\gather_x.npy')
    G38=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(der)\gather_x.npy')
    G39=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(izq)\gather_x.npy')
    G40=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=5 y e=0.7\gather_x.npy')
    G41=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=30 y e=1\gather_x.npy')
    G42=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\abajo-completo\a=15 y e=0.8\gather_x.npy')


    G43=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\modelo-base\gather_x.npy')
    G44=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(der)\gather_x.npy')
    G45=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(izq)\gather_x.npy')
    G46=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-completo\a=10 y e=0.7\gather_x.npy')
    G47=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\arriba-completo\gather_x.npy')
    G48=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\abajo-completo\a=10 y e=0.7\gather_x.npy')


    G49=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\Modelo_base\gather_x.npy')
    G50=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(cen)\gather_x.npy')
    G51=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(der)\gather_x.npy')
    G52=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(izq)\gather_x.npy')
    G53=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-completo\a=10 y e=0.7\gather_x.npy')
    G54=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=5 y e=0.3\gather_x.npy')
    G55=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=10 y e=0.7\gather_x.npy')
    G56=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=30 y e=0.3\gather_x.npy')

    G57=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\Modelo_base\gather_x.npy')
    G58=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(cen)\gather_x.npy')
    G59=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(der)\gather_x.npy')
    G60=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(izq)\gather_x.npy')
    G61=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=1 y e=0.3\gather_x.npy')
    G62=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=50 y e=0.3\gather_x.npy')
    G63=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=0.3\gather_x.npy')
    G64=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=1.4\gather_x.npy')
    G65=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=3\gather_x.npy')
    G66=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=0.3\gather_x.npy')
    G67=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=1.4\gather_x.npy')
    G68=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=3\gather_x.npy')
    G69=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=1 y e=0.3\gather_x.npy')
    G70=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=50 y e=0.3\gather_x.npy')

    G71=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\modelo base\gather_x.npy')
    G72=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(cen)\gather_x.npy')
    G73=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(der)\gather_x.npy')
    G74=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(izq)\gather_x.npy')
    G75=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=10 y e=0.6\gather_x.npy')
    G76=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=50 y e=0.6\gather_x.npy')
    G77=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(der)\gather_x.npy')
    G78=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(izq)\gather_x.npy')
    G79=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=1(der)\gather_x.npy')
    G80=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-completo\a=10 y e=0.3\gather_x.npy')
    G81=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=10 y e=0.6\gather_x.npy')
    G82=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=50 y e=0.6\gather_x.npy')


    G83=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\modelo base\gather_x.npy')
    G84=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(cen)\gather_x.npy')
    G85=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(der)\gather_x.npy')
    G86=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(izq)\gather_x.npy')
    G87=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-der)\gather_x.npy')
    G88=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-izq)\gather_x.npy')
    G89=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=2 y e=0.7\gather_x.npy')
    G90=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=25 y e=1\gather_x.npy')
    G91=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(der)\gather_x.npy')
    G92=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(izq)\gather_x.npy')
    G93=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(der)\gather_x.npy')
    G94=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(izq)\gather_x.npy')
    G95=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=0.5\gather_x.npy')
    G96=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=1.6\gather_x.npy')
    G97=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=0.5\gather_x.npy')
    G98=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=1.6\gather_x.npy')
    G99=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=10 y e=0.5\gather_x.npy')
    G100=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=100 y e=0.5\gather_x.npy')


    G101=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\base\bgather_x.npy')
    G102=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\a\gather_x.npy')
    G103=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\b\gather_x.npy')
    G104=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\c\gather_x.npy')
    G105=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\d\gather_x.npy')
    G106=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\e\gather_x.npy')
    G107=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\f\gather_x.npy')
    G108=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\g\gather_x.npy')
    G109=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\h\gather_x.npy')
    G110=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\a\gather_x.npy')
    G111=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\b\gather_x.npy')    
    G112=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\c\gather_x.npy')
    G113=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\d\gather_x.npy')
    G114=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\e\gather_x.npy')
    G115=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\f\gather_x.npy')
    G116=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\g\gather_x.npy')
    G117=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\h\gather_x.npy')
    G118=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\i\gather_x.npy')
    G119=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\a\gather_x.npy')
    G120=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\b\gather_x.npy')
    G121=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\c\gather_x.npy')    
    G122=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\d\gather_x.npy')
    G123=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\e\gather_x.npy')
    G124=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\f\gather_x.npy')
    G125=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\g\gather_x.npy')
    G126=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\h\gather_x.npy')
    G127=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\i\gather_x.npy')



    (Nt,Nx)=G1.shape
    G=np.zeros((127,Nt,Nx))

        
    G[0,:,:]=G1
    G[1,:,:]=G2
    G[2,:,:]=G3
    G[3,:,:]=G4
    G[4,:,:]=G5
    G[5,:,:]=G6
    G[6,:,:]=G7
    G[7,:,:]=G8
    G[8,:,:]=G9
    G[9,:,:]=G10
    G[10,:,:]=G11
    
    G[11,:,:]=G12
    G[12,:,:]=G13
    G[13,:,:]=G14
    G[14,:,:]=G15
    G[15,:,:]=G16
    G[16,:,:]=G17
    G[17,:,:]=G18
    G[18,:,:]=G19
    G[19,:,:]=G20
    
    G[20,:,:]=G21
    G[21,:,:]=G22
    G[22,:,:]=G23
    G[23,:,:]=G24
    
    G[24,:,:]=G25
    G[25,:,:]=G26
    G[26,:,:]=G27
    G[27,:,:]=G28
    G[28,:,:]=G29
    G[29,:,:]=G30
    G[30,:,:]=G31
    G[31,:,:]=G32
    
    G[32,:,:]=G33
    G[33,:,:]=G34
    G[34,:,:]=G35
    G[35,:,:]=G36
    G[36,:,:]=G37
    G[37,:,:]=G38
    G[38,:,:]=G39
    G[39,:,:]=G40
    G[40,:,:]=G41
    G[41,:,:]=G42
    
    G[42,:,:]=G43
    G[43,:,:]=G44
    G[44,:,:]=G45
    G[45,:,:]=G46
    G[46,:,:]=G47
    G[47,:,:]=G48

    G[48,:,:]=G49
    G[49,:,:]=G50
    G[50,:,:]=G51
    G[51,:,:]=G52
    G[52,:,:]=G53
    G[53,:,:]=G54
    G[54,:,:]=G55
    G[55,:,:]=G56
    
    G[56,:,:]=G57
    G[57,:,:]=G58
    G[58,:,:]=G59
    G[59,:,:]=G60
    G[60,:,:]=G61
    G[61,:,:]=G62
    G[62,:,:]=G63
    G[63,:,:]=G64
    G[64,:,:]=G65
    G[65,:,:]=G66
    G[66,:,:]=G67
    G[67,:,:]=G68
    G[68,:,:]=G69
    G[69,:,:]=G70
    
    G[70,:,:]=G71
    G[71,:,:]=G72
    G[72,:,:]=G73
    G[73,:,:]=G74
    G[74,:,:]=G75
    G[75,:,:]=G76
    G[76,:,:]=G77
    G[77,:,:]=G78
    G[78,:,:]=G79
    G[79,:,:]=G80
    G[80,:,:]=G81
    G[81,:,:]=G82
    
    G[82,:,:]=G83
    G[83,:,:]=G84
    G[84,:,:]=G85
    G[85,:,:]=G86
    G[86,:,:]=G87
    G[87,:,:]=G88
    G[88,:,:]=G89
    G[89,:,:]=G90 
    G[90,:,:]=G91
    G[91,:,:]=G92
    G[92,:,:]=G93
    G[93,:,:]=G94
    G[94,:,:]=G95
    G[95,:,:]=G96
    G[96,:,:]=G97
    G[97,:,:]=G98
    G[98,:,:]=G99
    G[99,:,:]=G100
    
    G[100,:,:]=G101
    G[101,:,:]=G102
    G[102,:,:]=G103
    G[103,:,:]=G104
    G[104,:,:]=G105
    G[105,:,:]=G106
    G[106,:,:]=G107
    G[107,:,:]=G108
    G[108,:,:]=G109
    G[109,:,:]=G110
    G[110,:,:]=G111
    G[111,:,:]=G112
    G[112,:,:]=G113
    G[113,:,:]=G114
    G[114,:,:]=G115
    G[115,:,:]=G116
    G[116,:,:]=G117
    G[117,:,:]=G118
    G[118,:,:]=G119
    G[119,:,:]=G120
    G[120,:,:]=G121
    G[121,:,:]=G122
    G[122,:,:]=G123
    G[123,:,:]=G124
    G[124,:,:]=G125
    G[125,:,:]=G126
    G[126,:,:]=G127
    
    G=normalize_2d(G)
    
    return G


def cargar_datos_y():
    C1=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Modelo_base\agrupacion.npy')
    C2=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=10 y e=0.4\agrupacion.npy')
    C3=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=0.4\agrupacion.npy')
    C4=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Completo\a=50 y e=2\agrupacion.npy')
    C5=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=5 y e=0.6\agrupacion.npy')
    C6=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\centrado-medio\a=125 y e=0.2\agrupacion.npy')
    C7=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=10 y e=0.6\agrupacion.npy')
    C8=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\arriba-medio\a=50 y e=1\agrupacion.npy')
    C9=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-medio\a=15 y e=0.4\agrupacion.npy')
    C10=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=5 y e=0.1\agrupacion.npy')
    C11=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\abajo-completo\a=30 y e=0.3\agrupacion.npy')


    C12=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelo_base\Modelo_base\agrupacion.npy')
    C13=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=10 y e=0.4\agrupacion.npy')
    C14=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\completo\a=50 y e=0.8\agrupacion.npy')
    C15=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=1 y e=0.3\agrupacion.npy')
    C16=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\centrado-medio\a=30 y e=0.3\agrupacion.npy')
    C17=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (cen)\agrupacion.npy')
    C18=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (der)\agrupacion.npy')
    C19=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\arriba-medio\a=20 y e=0.9 (izq)\agrupacion.npy')
    C20=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera_capa_no_tan_contrastada\abajo-medio\a=25 y e=0.2\agrupacion.npy')


    C21=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\modelo_base\agrupacion.npy')
    C22=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\centrado-medio\a=50 y e=1.2\agrupacion.npy')
    C23=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (der)\agrupacion.npy')
    C24=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\numeros_aleatorios_0_1\arriba_medio\a=3 y e=1.2 (izq)\agrupacion.npy')


    C25=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\modelo_base\agrupacion.npy')
    C26=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\centrado-medio\a=30 y e=0.7\agrupacion.npy')
    C27=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (der)\agrupacion.npy')
    C28=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba-medio\a=1 y e=1.4 (izq)\agrupacion.npy')
    C29=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\arriba_completo\agrupacion.npy')
    C30=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(der)\agrupacion.npy')
    C31=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-medio\a=10 y e=0.7(izq)\agrupacion.npy')
    C32=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambios_valores_de_capas_manuales\abajo-completo\a=10 y e=0.7\agrupacion.npy')


    C33=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\modelo-base\agrupacion.npy')
    C34=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(cen)\agrupacion.npy')
    C35=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(der)\agrupacion.npy')
    C36=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\medio-medio\a=25 y e=0.8(izq)\agrupacion.npy')
    C37=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(cen)\agrupacion.npy')
    C38=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(der)\agrupacion.npy')
    C39=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-medio\a=5 y e=0.7(izq)\agrupacion.npy')
    C40=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=5 y e=0.7\agrupacion.npy')
    C41=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\arriba-completo\a=30 y e=1\agrupacion.npy')
    C42=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\primera capa mas profunda\abajo-completo\a=15 y e=0.8\agrupacion.npy')


    C43=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\modelo-base\agrupacion.npy')
    C44=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(der)\agrupacion.npy')
    C45=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-medio\a=10 y e=0.7(izq)\agrupacion.npy')
    C46=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\medio-completo\a=10 y e=0.7\agrupacion.npy')
    C47=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\arriba-completo\agrupacion.npy')
    C48=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\capas iguales de profundas\abajo-completo\a=10 y e=0.7\agrupacion.npy')


    C49=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\Modelo_base\agrupacion.npy')
    C50=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(cen)\agrupacion.npy')
    C51=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(der)\agrupacion.npy')
    C52=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-medio\a=10 y e=0.7(izq)\agrupacion.npy')
    C53=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\mitad-completo\a=10 y e=0.7\agrupacion.npy')
    C54=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=5 y e=0.3\agrupacion.npy')
    C55=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=10 y e=0.7\agrupacion.npy')
    C56=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm mas profundo\arriba_completo\a=30 y e=0.3\agrupacion.npy')


    C57=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\Modelo_base\agrupacion.npy')
    C58=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(cen)\agrupacion.npy')
    C59=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(der)\agrupacion.npy')
    C60=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-medio\a=1 y e=0.8(izq)\agrupacion.npy')
    C61=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=1 y e=0.3\agrupacion.npy')
    C62=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\medio-completo\a=50 y e=0.3\agrupacion.npy')
    C63=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=0.3\agrupacion.npy')
    C64=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=1.4\agrupacion.npy')
    C65=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=1 y e=3\agrupacion.npy')
    C66=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=0.3\agrupacion.npy')
    C67=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=1.4\agrupacion.npy')
    C68=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\arriba completo\a=50 y e=3\agrupacion.npy')
    C69=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=1 y e=0.3\agrupacion.npy')
    C70=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\rm menos profundo\abajo-completo\a=50 y e=0.3\agrupacion.npy')


    C71=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\modelo base\agrupacion.npy')
    C72=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(cen)\agrupacion.npy')
    C73=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(der)\agrupacion.npy')
    C74=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-medio\a=10 y e=0.6(izq)\agrupacion.npy')
    C75=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=10 y e=0.6\agrupacion.npy')
    C76=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\medio-completo\a=50 y e=0.6\agrupacion.npy')
    C77=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(der)\agrupacion.npy')
    C78=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=0.3(izq)\agrupacion.npy')
    C79=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-medio\a=10 y e=1(der)\agrupacion.npy')
    C80=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\arriba-completo\a=10 y e=0.3\agrupacion.npy')
    C81=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=10 y e=0.6\agrupacion.npy')
    C82=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manual-rm mas profundo\abajo-completo\a=50 y e=0.6\agrupacion.npy')


    C83=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\modelo base\agrupacion.npy')
    C84=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(cen)\agrupacion.npy')
    C85=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(der)\agrupacion.npy')
    C86=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(izq)\agrupacion.npy')
    C87=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-der)\agrupacion.npy')
    C88=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-medio\a=10 y e=0.9(med-izq)\agrupacion.npy')
    C89=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=2 y e=0.7\agrupacion.npy')
    C90=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\medio-completo\a=25 y e=1\agrupacion.npy')
    C91=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(der)\agrupacion.npy')
    C92=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=0.5(izq)\agrupacion.npy')
    C93=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(der)\agrupacion.npy')
    C94=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-medio\a=10 y e=1.6(izq)\agrupacion.npy')
    C95=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=0.5\agrupacion.npy')
    C96=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=10 y e=1.6\agrupacion.npy')
    C97=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=0.5\agrupacion.npy')
    C98=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\arriba-completo\a=100 y e=1.6\agrupacion.npy')
    C99=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=10 y e=0.5\agrupacion.npy')
    C100=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\cambio manuel-rm menos profundo\abajo-completo\a=100 y e=0.5\agrupacion.npy')


    C101=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\base\agrupacion.npy')
    C102=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\a\agrupacion.npy')
    C103=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\b\agrupacion.npy')
    C104=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\c\agrupacion.npy')
    C105=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\d\agrupacion.npy')
    C106=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\e\agrupacion.npy')
    C107=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\f\agrupacion.npy')
    C108=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\g\agrupacion.npy')
    C109=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\a\h\agrupacion.npy')
    C110=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\a\agrupacion.npy')
    C111=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\b\agrupacion.npy')    
    C112=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\c\agrupacion.npy')
    C113=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\d\agrupacion.npy')
    C114=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\e\agrupacion.npy')
    C115=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\f\agrupacion.npy')
    C116=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\g\agrupacion.npy')
    C117=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\h\agrupacion.npy')
    C118=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\b\i\agrupacion.npy')
    C119=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\a\agrupacion.npy')
    C120=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\b\agrupacion.npy')
    C121=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\c\agrupacion.npy')    
    C122=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\d\agrupacion.npy')
    C123=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\e\agrupacion.npy')
    C124=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\f\agrupacion.npy')
    C125=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\g\agrupacion.npy')
    C126=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\h\agrupacion.npy')
    C127=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos\modelos nuevos\1\c\i\agrupacion.npy')





    C1=atributo_rm(C1)
    C2=atributo_rm(C2)
    C3=atributo_rm(C3)
    C4=atributo_rm(C4)
    C5=atributo_rm(C5)
    C6=atributo_rm(C6)
    C7=atributo_rm(C7)
    C8=atributo_rm(C8)
    C9=atributo_rm(C9)
    C10=atributo_rm(C10)
    C11=atributo_rm(C11)
    
    C12=atributo_rm(C12)
    C13=atributo_rm(C13)
    C14=atributo_rm(C14)
    C15=atributo_rm(C15)
    C16=atributo_rm(C16)
    C17=atributo_rm(C17)
    C18=atributo_rm(C18)
    C19=atributo_rm(C19)
    C20=atributo_rm(C20)
    
    C21=atributo_rm(C21)
    C22=atributo_rm(C22)
    C23=atributo_rm(C23)
    C24=atributo_rm(C24)
    
    C25=atributo_rm(C25)
    C26=atributo_rm(C26)
    C27=atributo_rm(C27)
    C28=atributo_rm(C28)
    C29=atributo_rm(C29)
    C30=atributo_rm(C30)
    C31=atributo_rm(C31)
    C32=atributo_rm(C32)
    
    C33=atributo_rm(C33)
    C34=atributo_rm(C34)
    C35=atributo_rm(C35)
    C36=atributo_rm(C36)
    C37=atributo_rm(C37)
    C38=atributo_rm(C38)
    C39=atributo_rm(C39)
    C40=atributo_rm(C40)
    C41=atributo_rm(C41)
    C42=atributo_rm(C42)
    
    C43=atributo_rm(C43)
    C44=atributo_rm(C44)
    C45=atributo_rm(C45)
    C46=atributo_rm(C46)
    C47=atributo_rm(C47)
    C48=atributo_rm(C48)
    
    C49=atributo_rm(C49)
    C50=atributo_rm(C50)
    C51=atributo_rm(C51)
    C52=atributo_rm(C52)
    C53=atributo_rm(C53)
    C54=atributo_rm(C54)
    C55=atributo_rm(C55)
    C56=atributo_rm(C56)

    C57=atributo_rm(C57)
    C58=atributo_rm(C58)
    C59=atributo_rm(C59)
    C60=atributo_rm(C60)
    C61=atributo_rm(C61)
    C62=atributo_rm(C62)
    C63=atributo_rm(C63)
    C64=atributo_rm(C64)
    C65=atributo_rm(C65)
    C66=atributo_rm(C66)
    C67=atributo_rm(C67)
    C68=atributo_rm(C68)
    C69=atributo_rm(C69)
    C70=atributo_rm(C70)
    
    C71=atributo_rm(C71)
    C72=atributo_rm(C72)
    C73=atributo_rm(C73)
    C74=atributo_rm(C74)
    C75=atributo_rm(C75)
    C76=atributo_rm(C76)
    C77=atributo_rm(C77)
    C78=atributo_rm(C78)
    C79=atributo_rm(C79)
    C80=atributo_rm(C80)
    C81=atributo_rm(C81)
    C82=atributo_rm(C82)

    C83=atributo_rm(C83)
    C84=atributo_rm(C84)
    C85=atributo_rm(C85)
    C86=atributo_rm(C86)
    C87=atributo_rm(C87)
    C88=atributo_rm(C88)
    C89=atributo_rm(C89)
    C90=atributo_rm(C90)
    C91=atributo_rm(C91)
    C92=atributo_rm(C92)
    C93=atributo_rm(C93)
    C94=atributo_rm(C94)
    C95=atributo_rm(C95)
    C96=atributo_rm(C96)
    C97=atributo_rm(C97)
    C98=atributo_rm(C98)
    C99=atributo_rm(C99)
    C100=atributo_rm(C100)


    C101=atributo_rm(C101)
    C102=atributo_rm(C102)
    C103=atributo_rm(C103)
    C104=atributo_rm(C104)
    C105=atributo_rm(C105)
    C106=atributo_rm(C106)
    C107=atributo_rm(C107)
    C108=atributo_rm(C108)
    C109=atributo_rm(C109)
    C110=atributo_rm(C110)
    C111=atributo_rm(C111)
    
    C112=atributo_rm(C112)
    C113=atributo_rm(C113)
    C114=atributo_rm(C114)
    C115=atributo_rm(C115)
    C116=atributo_rm(C116)
    C117=atributo_rm(C117)
    C118=atributo_rm(C118)
    C119=atributo_rm(C119)
    C120=atributo_rm(C120)
    C121=atributo_rm(C121)
    C122=atributo_rm(C122)
    C123=atributo_rm(C123)
    C124=atributo_rm(C124)
    C125=atributo_rm(C125)
    C126=atributo_rm(C126)
    C127=atributo_rm(C127)
    

    (Nt,Nx)=(C1.shape[0],C1.shape[1])
    C=np.zeros((127,Nt,Nx))

    C[0,:,:]=C1
    C[1,:,:]=C2
    C[2,:,:]=C3
    C[3,:,:]=C4
    C[4,:,:]=C5
    C[5,:,:]=C6
    C[6,:,:]=C7
    C[7,:,:]=C8
    C[8,:,:]=C9
    C[9,:,:]=C10
    C[10,:,:]=C11
    
    C[11,:,:]=C12
    C[12,:,:]=C13
    C[13,:,:]=C14
    C[14,:,:]=C15
    C[15,:,:]=C16
    C[16,:,:]=C17
    C[17,:,:]=C18
    C[18,:,:]=C19
    C[19,:,:]=C20
    
    C[20,:,:]=C21
    C[21,:,:]=C22
    C[22,:,:]=C23
    C[23,:,:]=C24
    
    C[24,:,:]=C25
    C[25,:,:]=C26
    C[26,:,:]=C27
    C[27,:,:]=C28
    C[28,:,:]=C29
    C[29,:,:]=C30
    C[30,:,:]=C31
    C[31,:,:]=C32
    
    C[32,:,:]=C33
    C[33,:,:]=C34
    C[34,:,:]=C35
    C[35,:,:]=C36
    C[36,:,:]=C37
    C[37,:,:]=C38
    C[38,:,:]=C39
    C[39,:,:]=C40
    C[40,:,:]=C41
    C[41,:,:]=C42
    
    C[42,:,:]=C43
    C[43,:,:]=C44
    C[44,:,:]=C45
    C[45,:,:]=C46
    C[46,:,:]=C47
    C[47,:,:]=C48
    
    C[48,:,:]=C49
    C[49,:,:]=C50
    C[50,:,:]=C51
    C[51,:,:]=C52
    C[52,:,:]=C53
    C[53,:,:]=C54
    C[54,:,:]=C55
    C[55,:,:]=C56
    
    
    C[56,:,:]=C57
    C[57,:,:]=C58
    C[58,:,:]=C59
    C[59,:,:]=C60
    C[60,:,:]=C61
    C[61,:,:]=C62
    C[62,:,:]=C63
    C[63,:,:]=C64
    C[64,:,:]=C65
    C[65,:,:]=C66
    C[66,:,:]=C67
    C[67,:,:]=C68
    C[68,:,:]=C69
    C[69,:,:]=C70
    
    
    C[70,:,:]=C71
    C[71,:,:]=C72
    C[72,:,:]=C73
    C[73,:,:]=C74
    C[74,:,:]=C75
    C[75,:,:]=C76  
    C[76,:,:]=C77
    C[77,:,:]=C78
    C[78,:,:]=C79
    C[79,:,:]=C80
    C[80,:,:]=C81
    C[81,:,:]=C82
    
    C[82,:,:]=C83
    C[83,:,:]=C84
    C[84,:,:]=C85
    C[85,:,:]=C86
    C[86,:,:]=C87
    C[87,:,:]=C88
    C[88,:,:]=C89
    C[89,:,:]=C90
    C[90,:,:]=C91
    C[91,:,:]=C92
    C[92,:,:]=C93
    C[93,:,:]=C94
    C[94,:,:]=C95
    C[95,:,:]=C96  
    C[96,:,:]=C97
    C[97,:,:]=C98
    C[98,:,:]=C99
    C[99,:,:]=C100
    
    C[100,:,:]=C101
    C[101,:,:]=C102
    C[102,:,:]=C103
    C[103,:,:]=C104
    C[104,:,:]=C105
    C[105,:,:]=C106
    C[106,:,:]=C107
    C[107,:,:]=C108
    C[108,:,:]=C109
    C[109,:,:]=C110
    C[110,:,:]=C111
    C[111,:,:]=C112
    C[112,:,:]=C113
    C[113,:,:]=C114
    C[114,:,:]=C115
    C[115,:,:]=C116
    C[116,:,:]=C117
    C[117,:,:]=C118
    C[118,:,:]=C119
    C[119,:,:]=C120
    C[120,:,:]=C121
    C[121,:,:]=C122
    C[122,:,:]=C123
    C[123,:,:]=C124
    C[124,:,:]=C125
    C[125,:,:]=C126
    C[126,:,:]=C127
    
    return C