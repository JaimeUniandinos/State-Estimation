import  numpy as np
import math
import sys
import cmath

# numero de buses 
#         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
#         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata6 =[[1, 2, complex(0.1, 0.2),    complex(0, 0.02), 1],
            [1, 4, complex(0.05, 0.2),   complex(0, 0.02), 1],
            [1, 5, complex(0.08, 0.3),   complex(0, 0.03), 1],
            [2, 3, complex(0.05, 0.025), complex(0, 0.03), 1],
            [2, 4, complex(0.05, 0.01),  complex(0, 0.01), 1],
            [2, 5, complex(0.01, 0.3),   complex(0, 0.02), 1],
            [2, 6, complex(0.07, 0.2),   complex(0, 0.025), 1],
            [3, 5, complex(0.12, 0.26),  complex(0, 0.025), 1],
            [3, 6, complex(0.02, 0.1),   complex(0, 0.01), 1],
            [4, 5, complex(0.2, 0.4),    complex(0, 0.04), 1],
            [5, 6, complex(0.1, 0.03),   complex(0, 0.03), 1]]


linedata9 =[[1, 4, complex(0.00, 0.0576),    complex(0, 0.00), 1],
            [4, 5, complex(0.017, 0.092),   complex(0, 0.079), 1],
            [5, 6, complex(0.039, 0.170),   complex(0, 0.179), 1],
            [3, 6, complex(0.0, 0.0586), complex(0, 0.0), 1],
            [6, 7, complex(0.0119, 0.1008),  complex(0, 0.1045), 1],
            [7, 8, complex(0.0085, 0.072),   complex(0, 0.0745), 1],
            [8, 2, complex(0.0, 0.0625),   complex(0, 0.0), 1],
            [8, 9, complex(0.032, 0.161),  complex(0, 0.153), 1],
            [9, 4, complex(0.01, 0.085),   complex(0, 0.088), 1]]


def ybusppg(num):
    fb = np.array(linedata9)[:,0]      # From bus number...
    tb = np.array(linedata9)[:,1]      # To bus number...
    z = np.array(linedata9)[:,2]       # Z matrix....
    b = np.array(linedata9)[:,3]       # Ground Admittance, B/2...
    a = np.array(linedata9)[:,4]       # Tap setting value..
    y=1/z                               # To get inverse of each element...  
                                        
    nbus = 9                           # number of buses
    nbrach= len(fb)                     # no. of branches..
    
    ybus= np.zeros((nbus,nbus),dtype = 'complex_') # Initialise Ybus
    #Formation of the Off Diagonal Elements...
    for k in range(nbrach):
        ybus[int(fb[k])-1,int(tb[k])-1]= ybus[int(fb[k])-1,int(tb[k])-1]-(y[k]/a[k])
        ybus[int(tb[k])-1,int(fb[k])-1]= ybus[int(fb[k])-1,int(tb[k])-1]
    

    #Formation of the Off Diagonal Elements...
    for m in range(nbus):
        for n in range(nbrach):
            

            if int(fb[n]-1)==m:
                prueba=y[n]/(a[n]**2)+b[n]
                ybus[m,m]=ybus[m,m]+ prueba

            elif int(tb[n]-1) == m:
                ybus[m,m]=ybus[m,m]+ y[n] + b[n]
    #ybus                  % Bus Admittance Matrix
    #zbus = inv(ybus)      % Bus Impedance Matrix
    return ybus




def bbusppg(num): # Line Data for B-Bus (Shunt Admittance)Formation.

    fb = np.array(linedata9)[:,0]          # From bus number...
    tb = np.array(linedata9)[:,1]          # To bus number...
    b = np.array(linedata9)[:,3]           # Ground Admittance, B/2...
    nbus = int(max(max(fb), max(tb)))       # no. of buses...
    nbrach = len(fb)                        # no. of branches...
    
    bbus = np.zeros((nbus,nbus),dtype = 'complex_')
    for k in range(nbrach):
        bbus[int(fb[k])-1,int(tb[k])-1] = b[k]
        bbus[int(tb[k])-1,int(fb[k])-1] = bbus[int(fb[k])-1,int(tb[k])-1]
    return bbus

#% IEEE - 6...
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01
wct41=0.01

num=9
ybus = ybusppg(num)                     # Get YBus..
bpq = bbusppg(num)                      # Get B data..
nbus = 9                               # Get number of buses..
MatrizA=[]
pin1=1
pin2=1
pin3=1

pmu=[]
pines=[pin1,pin2,pin3]
pinesvalues=[4,6,8]
for i in range(len(pines)):
    if pines[i]==1:
        pmu.append(pinesvalues[i])


for k in range(len(pmu)):
    for i in range(len(linedata9)):
        h4 = np.zeros((nbus))
        if linedata9[i][1]==pmu[k]:
            h4[(linedata9[i][0])-1]=-1
            h4[(linedata9[i][1])-1]=1
            MatrizA.append(h4)
        if linedata9[i][0]==pmu[k]:
            h4[(linedata9[i][0])-1]=1
            h4[(linedata9[i][1])-1]=-1
            MatrizA.append(h4) #LA matriz A (MAtriz de incidencia)es una matriz que nos permite saber cuales lineas están contectadas,
                               #es decir cada fila de esta matriz es una linea y cada columna es un nodo

Y = np.zeros((len(MatrizA),len(MatrizA)),dtype = 'complex_') #Matriz de ceros con dimensiones n,n donde n= el numero de lineas
YS = np.zeros((len(MatrizA),nbus),dtype = 'complex_')

for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
         
        if MatrizA[i][k]==-1:
            y=k
    Y[i,i]=ybus[x,y] # Se crea la matriz Admitancia en serie para las lineas de entrada y de salida
    YS[i,x]=bpq[x,y] # Se crea la matriz admitancia shunt para las lineas de entrada y de salida

II=[]
for i in range(len(pmu)):
    ii=[]
    for j in range(nbus):
        ii.append(0 if j!=pmu[i]-1 else 1)
    II.append(ii) #Matriz con la información de los nodos que tienen PMU

M= np.dot(Y,MatrizA)
for a in range(len(MatrizA)):
    for b in range(nbus):
       M[a][b]=M[a][b]+YS[a][b]
II=np.asarray(II)
B=[]
for i in range(II.shape[0]):
    row=[]
    for k in range(II.shape[1]):
        row.append(II[i,k])    
    B.append(row)
    
for i in range(M.shape[0]):
    row=[]
    for k in range(M.shape[1]):
        row.append(M[i,k])    
    B.append(row)
B=np.asarray(B)



IN=[[1,4,complex(2.576331*math.cos(math.radians(-20.678366)), 2.576331*math.sin(math.radians(-20.678366))),16.5],#Trafo1 Primero---
    [4,1,complex(0.184824*math.cos(math.radians(159.321634)), 0.184824*math.sin(math.radians(159.321634))),230],#Transformador Nodo 1 Base de Voltaje 16.5KV  
    [4,5,complex(0.14152*math.cos(math.radians(132.449057)), 0.14152*math.sin(math.radians(132.449057))),230],#Segundo--
    [5,4,complex(0.114761*math.cos(math.radians(-31.429649)),0.114761*math.sin(math.radians(-31.429649))),230],
    [5,6,complex(0.214509*math.cos(math.radians(168.367685)),0.214509*math.sin(math.radians(168.367685))),230],
    [6,5,complex(0.212976*math.cos(math.radians(9.245459)),0.212976*math.sin(math.radians(9.245459))),230],#Tercero----
    [4,9,complex(0.086102*math.cos(math.radians(147.87046)),0.086102*math.sin(math.radians(147.87046))),230],
    [9,4,complex(0.075189*math.cos(math.radians(-4.136061)),0.075189*math.sin(math.radians(-4.136061))),230],
    [3,6,complex(5.104956*math.cos(math.radians(6.941622)), 5.104956*math.sin(math.radians(6.941622))),13.8],#Trafo  18KV
    [6,3,complex(0.399518*math.cos(math.radians(-173.058378)), 0.399518*math.sin(math.radians(-173.058378))),230],#Transformador Nodo 2
    [6,7,complex(0.186911*math.cos(math.radians(4.316299)),0.186911*math.sin(math.radians(4.316299))),230],
    [7,6,complex(0.189402*math.cos(math.radians(172.6984)),0.189402*math.sin(math.radians(172.6984))),230],
    [7,8,complex(0.084559*math.cos(math.radians(135.496321)),0.084559*math.sin(math.radians(135.496321))),230],
    [8,7,complex(0.059303*math.cos(math.radians(-5.381918)),0.059303*math.sin(math.radians(-5.381918))),230],
    [2,8,complex(3.497615*math.cos(math.radians(11.94555)), 3.497615*math.sin(math.radians(11.94555))),18],#Trafo3 13.8KV
    [8,2,complex(0.209857*math.cos(math.radians(-168.05445)), 0.209857*math.sin(math.radians(-168.05445))),230],#Transformador Nodo 3
    [8,9,complex(0.151114*math.cos(math.radians(163.559563)),0.151114*math.sin(math.radians(163.559563))),230],
    [9,8,complex(0.15426*math.cos(math.radians(18.520166)),0.15426*math.sin(math.radians(18.520166))),230]]



def CorrienteBase(VoltajeBase):
    corrientebase=100/(math.sqrt(3)*VoltajeBase)#100MVA/raiz(3)*VoltajeBase
    return corrientebase


Est2 = np.zeros((len(MatrizA)+(len(pmu))),dtype = 'complex_')
for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
        if MatrizA[i][k]==-1:
            y=k
    for j in range (len(IN)):
        if(IN[j][0]==y+1 and IN[j][1]==x+1):
            Est2[i+len(pmu)]=IN[j][2]/CorrienteBase(IN[j][3])
            break
VN=[ complex(1.040*math.cos(math.radians(0)),1.040*math.sin(math.radians(0))),
     complex(1.025*math.cos(math.radians(9.280)),1.025*math.sin(math.radians(9.280))),#
     complex(1.025*math.cos(math.radians(4.665)),1.025*math.sin(math.radians(4.665))),
     complex(1.026*math.cos(math.radians(-2.217)),1.026*math.sin(math.radians(-2.217))),#
     complex(0.996*math.cos(math.radians(-3.988)),0.996*math.sin(math.radians(-3.988))),#
     complex(1.013*math.cos(math.radians(-3.688)),1.013*math.sin(math.radians(-3.688))),
     complex(1.026*math.cos(math.radians(3.719)),1.026*math.sin(math.radians(3.719))),
     complex(1.016*math.cos(math.radians(0.726)),1.016*math.sin(math.radians(0.726))),
     complex(1.0323*math.cos(math.radians(1.966)),1.0323*math.sin(math.radians(1.966)))]




Est2 = np.zeros((len(MatrizA)+(len(pmu))),dtype = 'complex_')
for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
        if MatrizA[i][k]==-1:
            y=k
    for t in range(len(linedata9)):
        if (linedata9[t][0]==x+1 or linedata9[t][1]==x+1) and (linedata9[t][0]==y+1 or linedata9[t][1]==y+1):
            f=t
            break

    Est2[i+len(pmu)]=((VN[y]-VN[x])/(linedata9[f][2]))

VN=np.asarray(VN)
for i in range(len(pmu)):
        Est2[i]=VN[pmu[i]-1]

W = np.zeros(((len(Est2)),len(Est2)))

for i in range(len(Est2)):
    if i<len(pmu):
        W[i,i]=1/(0.01**2)
    else:
        W[i,i]=1/(0.01**2)

Xest=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),Est2)

XWest=np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(B),W),B)),np.dot(np.transpose(B),W)),Est2)

Bt=B.transpose()#B transpuesta
BtWB=np.linalg.inv(np.dot(np.dot(Bt,W),B))

Xpolar=[]
for i in range(len(Xest)):
    Xpolar.append([cmath.polar(Xest[i])[0],math.degrees(cmath.polar(Xest[i])[1])])

Error=np.zeros((nbus),dtype = 'complex_')
for i in range(len(Xest)):
    Error[i]=100*abs(Xest[i]-VN[i])/VN[i]
    Error[i]= math.sqrt(Error[i].real**2+Error[i].imag**2)

ErrorW=np.zeros((nbus),dtype = 'complex_')
for i in range(len(XWest)):
    ErrorW[i]=100*abs(XWest[i]-VN[i])/VN[i] 
    ErrorW[i]= math.sqrt(ErrorW[i].real**2+ErrorW[i].imag**2)

print('Xpolares:')
print(Xpolar)
corrientesEstimadas=np.zeros(len(MatrizA),dtype = 'complex_')
for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
        if MatrizA[i][k]==-1:
            y=k
    for t in range(len(linedata9)):
        if (linedata9[t][0]==x+1 or linedata9[t][1]==x+1) and (linedata9[t][0]==y+1 or linedata9[t][1]==y+1):
            Bmedios=-linedata9[t][3]
            f=t
            
    
    corrientesEstimadas[i]=((Xest[y]-Xest[x])/(linedata9[f][2]))-(Bmedios*Xest[x])#Se calcula la corriente Estimada para cada una de las linea  
    

VectorObservador=[]
for i in range(len(pinesvalues)):
    VectorObservador.append(cmath.polar((Est2[i]-Xest[pinesvalues[i]-1]))[0]/0.01)
for i in range(len(corrientesEstimadas)):
    VectorObservador.append(cmath.polar((Est2[len(pinesvalues)+i]-corrientesEstimadas[i]))[0]/0.01)# Genera el vector de observación completo[medidasVoltaje,MedidasCorriente]

print('Est2')
print(Est2)
print('corrientesEstimadas')
print(corrientesEstimadas)



print('VectorObservador')
print(VectorObservador)

