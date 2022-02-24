import numpy as np
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


def ybusppg(num):
    fb = np.array(linedata6)[:,0]      # From bus number...
    tb = np.array(linedata6)[:,1]      # To bus number...
    z = np.array(linedata6)[:,2]       # Z matrix....
    b = np.array(linedata6)[:,3]       # Ground Admittance, B/2...
    a = np.array(linedata6)[:,4]       # Tap setting value..
    y=1/z                               # To get inverse of each element...  
                                        
    nbus = 6                           # number of buses
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

    fb = np.array(linedata6)[:,0]          # From bus number...
    tb = np.array(linedata6)[:,1]          # To bus number...
    b = np.array(linedata6)[:,3]           # Ground Admittance, B/2...
    nbus = int(max(max(fb), max(tb)))       # no. of buses...
    nbrach = len(fb)                        # no. of branches...
    
    bbus = np.zeros((nbus,nbus),dtype = 'complex_')
    for k in range(nbrach):
        bbus[int(fb[k])-1,int(tb[k])-1] = b[k]
        bbus[int(tb[k])-1,int(fb[k])-1] = bbus[int(fb[k])-1,int(tb[k])-1]
    return bbus

#% IEEE - 6...
num=6
ybus = ybusppg(num)                     # Get YBus..
bpq = bbusppg(num)                      # Get B data..
nbus = 6                               # Get number of buses..
MatrizA=[]
pin1=1
pin2=1
pin3=1

pmu=[]
pines=[pin1,pin2,pin3]
pinesvalues=[2,4,5]
for i in range(len(pines)):
    if pines[i]==1:
        pmu.append(pinesvalues[i])


for k in range(len(pmu)):
    for i in range(len(linedata6)):
        h4 = np.zeros((nbus))
        if linedata6[i][1]==pmu[k]:
            h4[(linedata6[i][0])-1]=-1
            h4[(linedata6[i][1])-1]=1
            MatrizA.append(h4)
        if linedata6[i][0]==pmu[k]:
            h4[(linedata6[i][0])-1]=1
            h4[(linedata6[i][1])-1]=-1
            MatrizA.append(h4)

Y = np.zeros((len(MatrizA),len(MatrizA)),dtype = 'complex_')
YS = np.zeros((len(MatrizA),nbus),dtype = 'complex_')

for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
         
        if MatrizA[i][k]==-1:
            y=k
    Y[i,i]=ybus[x,y]
    YS[i,x]=bpq[x,y]

II=[]
for i in range(len(pmu)):
    ii=[]
    for j in range(nbus):
        ii.append(0 if j!=pmu[i]-1 else 1)
    II.append(ii)

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


VN=[ complex(1.050*math.cos(math.radians(0)),1.050*math.sin(math.radians(0))),
     complex(1.050*math.cos(math.radians(-3.671)),1.050*math.sin(math.radians(-3.671))),#
     complex(1.070*math.cos(math.radians(-4.273)),1.050*math.sin(math.radians(-4.273))),
     complex(0.989*math.cos(math.radians(-4.129)),0.989*math.sin(math.radians(-4.129))),#
     complex(0.985*math.cos(math.radians(-5.276)),0.985*math.sin(math.radians(-5.276))),#
     complex(1.004*math.cos(math.radians(-5.947)),1.004*math.sin(math.radians(-5.947)))]

VN=np.asarray(VN)

Est = np.zeros((len(MatrizA)+(len(pmu))),dtype = 'complex_')
for i in range(len(MatrizA)):
    for k in range(nbus):
        if MatrizA[i][k]==1:
            x=k
        if MatrizA[i][k]==-1:
            y=k
    for t in range(len(linedata6)):
        if (linedata6[t][0]==x+1 or linedata6[t][1]==x+1) and (linedata6[t][0]==y+1 or linedata6[t][1]==y+1):
            f=t
    
    Est[i+len(pmu)]=(VN[y]-VN[x])/(linedata6[f][2])
for i in range(len(pmu)):
        Est[i]=VN[pmu[i]-1]

W = np.zeros(((len(Est)),len(Est)))
for i in range(len(Est)):
    if i<len(pmu):
        W[i,i]=0.01**2
    else:
        W[i,i]=0.01**2
Wi=np.linalg.inv(W)
Xest=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),Est)
XWest=np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(B),Wi),B)),np.dot(np.transpose(B),Wi)),Est)

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
print(Xest[2])
print('XWest:')
print(XWest)
print('VN:')
print(VN)
print('Error:')
print(Error.real)
print('ErrorW:')
print(ErrorW.real)
print('error')
print(max(Error))