import numpy as np
import math
import sys


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

zdata6   = [ #---- Voltage Magnitude ------------%
            [1,     1,    1.05,    1,   0,  9e-4],
            #-----------------------------------%
            #---- Real Power Injection ---------%
           [2,    2 ,   1.0788,     1,      0 ,  1e-4],
           [3,    2 ,   0.50 ,     2,      0 ,  1e-4],
           [4,    2 ,   0.60 ,     3,      0 ,  1e-4],
           [5,    2 ,  -0.70 ,     4,      0 ,  1e-4],
           [6,    2,   -0.70 ,     5,      0 ,  1e-4],

           #------------------------------------%
           #---- Reative Power Injection -------%
           [7,    3 ,  0.1596,     1,      0 ,  1e-4],
           [8,    3,   0.7436,     2,      0 ,  1e-4],
           [9,    3,   0.8963,     3,      0 ,  1e-4],
           [10,    3,   -0.70 ,     4,      0 ,  1e-4],
           [11,    3,   -0.70 ,     5,      0 ,  1e-4],

           #------------------------------------%
           #------ Real Power Flow ------------- %
           [12,     4,   -0.2778,   2,       1,  64e-6],
           [13,     4,    0.0293,   2,       3,  64e-6],
           [14,     4,    0.3309,   2,       4,  64e-6],
           [15,     4,    0.1551,   2,       5,  64e-6],
           [16,     4,    0.2625,   2,       6,  64e-6],
           [17,     4,    0.0408,   4,       5,  64e-6],
           [18,     4,   -0.1802,   5,       3,  64e-6],
           [19,     4,    0.0161,   5,       6,  64e-6],
           #----   --------------------------------%
           #------ Real Power Flow ------------- %
           [20,     5,    0.1282,   2,       1,  64e-6],
           [21,     5,   -0.1227,   2,       3,  64e-6],
           [22,     5,    0.4605,   2,       4,  64e-6],
           [23,     5,    0.1535,   2,       5,  64e-6],
           [24,     5,    0.1240,   2,       6,  64e-6],
           [25,     5,   -0.0494,   4,       5,  64e-6],
           [26,     5,   -0.2610,   5,       3,  64e-6],
           [27,     5,   -0.0966,   5,       6,  64e-6],]
           #--------------------------------------%


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
    return bbus.imag


#% IEEE - 6 or IEEE - 30 bus system..(for IEEE-14 bus system replace 30 by 14)...
num=6
ybus = ybusppg(num)                     # Get YBus..
bpq = bbusppg(num)                      # Get B data..
nbus = 6                               # Get number of buses..
typee = np.array(zdata6)[:,1]          # Type of measurement, Vi - 1, Pi - 2, Qi - 3, Pij - 4, Qij - 5, Iij - 6..
z = np.array(zdata6)[:,2]              # Measuement values..
fbus = np.array(zdata6)[:,3]           # From bus..
tbus = np.array(zdata6)[:,4]           # To bus..
Ri = np.diag(np.array(zdata6)[:,5])    # % Measurement Error..
V = np.ones((nbus,1))                   # Initialize the bus voltages..
dell = np.zeros((nbus,1))               # Initialize the bus angles..
E = [*dell[1:],*V]                      # State Vector..

G = ybus.real
B = ybus.imag
vi=[]
ppi=[]
qi=[]
pf=[]
qf=[]

for k in range(0,len(typee)):
    if typee[k]==1:                     # Index of voltage magnitude measurements..
        vi.append(k)
    elif typee[k]==2:                   # Index of real power injection measurements..
        ppi.append(k)
    elif typee[k]==3:                   # Index of reactive power injection measurements..
        qi.append(k)
    elif typee[k]==4:                   # Index of real powerflow measurements..
        pf.append(k)
    elif typee[k]==5:                   # Index of reactive powerflow measurements..
        qf.append(k)
nvi = len(vi)                           # Number of Voltage measurements..
npi = len(ppi)                          # Number of Real Power Injection measurements..
nqi = len(qi)                           # Number of Reactive Power Injection measurements..
npf = len(pf)                           # Number of Real Power Flow measurements..
nqf = len(qf)                           # Number of Reactive Power Flow measurements..

iter = 1
tol=1

while tol > 1e-4:
    #Measurement Function, h
    h=[]
    h1 = V[int(fbus[vi]-1),0]
    h2 = np.zeros((npi,1))
    h3 = np.zeros((nqi,1))
    h4 = np.zeros((npf,1))
    h5 = np.zeros((nqf,1)) 
    h.append(h1)

    for i in range(npi):
        m = int(fbus[ppi[i]])
        for k in range(nbus):
            a=V[m-1]
            b=V[k]
            c=G[m-1,k]*math.cos(dell[m-1]-dell[k])
            d=B[m-1,k]*math.sin(dell[m-1]-dell[k])
            h2[i] = h2[i] + V[m-1]*V[k]*(G[m-1,k]*math.cos(dell[m-1]-dell[k]) + B[m-1,k]*math.sin(dell[m-1]-dell[k]))
        h.append(h2[i])    
    
    print('h2:')
    print(h2)
    print('--------')
    tol=tol-1      
        
    for i in range(nqi):
        m = int(fbus[qi[i]])
        for k in range(nbus):
            h3[i] = h3[i] + V[m-1]*V[k]*(G[m-1,k]*math.sin(dell[m-1]-dell[k]) - B[m-1,k]*math.cos(dell[m-1]-dell[k]))
        h.append(h3[i])
    print('h3:')
    print(h3)
    print('--------')

    for i in range(npf):
        m = int(fbus[pf[i]])
        n = int(tbus[pf[i]])
        a= V[m-1]*V[m-1]*G[m-1,n-1]
        b= V[m-1]*V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1])- B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))
        h4[i] =-a-b
        h.append(h4[i])
    print('h4:')
    print(h4)
    print('--------')

    for i in range(nqf):
        m = int(fbus[qf[i]])
        n = int(tbus[qf[i]])
        a=V[m-1]*V[m-1]*(-B[m-1,n-1]+bpq[m-1,n-1])
        b=V[m-1]*V[n-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1])+ B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]))
        h5[i] = -a-b
        h.append(h5[i])
    print('h5:')
    print(h5)
    print('--------')

    #h=[h1,h2,h3,h4,h5]
    
    #Residue..
    r= z-h

    # Jacobian..
    # H11 - Derivative of V with respect to angles.. All Zeros
    H11 = np.zeros((nvi,nbus-1))


    # H12 - Derivative of V with respect to V..    
    H12 = np.zeros((nvi,nbus))
    for k in range(nvi):
        for n in range(nbus):
            if n == k:
                H12[k,n] = 1


    # H21 - Derivative of Real Power Injections with Angles..
    H21 = np.zeros((npi,nbus-1))
    for i in range(npi):
        m = int(fbus[ppi[i]])
        for k in range(nbus-1):
            if k+1 == m-1:
                for n in range (nbus):
                    H21[i,k] = H21[i,k] + V[m-1]* V[n]*(-G[m-1,n]*math.sin(dell[m-1]-dell[n]) + B[m-1,n]*math.cos(dell[m-1]-dell[n]))
                
                H21[i,k] = H21[i,k] - V[m-1]*V[m-1]*B[m-1,m-1]
            else:
                H21[i,k] = V[m-1]* V[k+1]*(G[m-1,k+1]*math.sin(dell[m-1]-dell[k+1]) - B[m-1,k+1]*math.cos(dell[m-1]-dell[k+1]))



    # H22 - Derivative of Real Power Injections with V..
    H22 = np.zeros((npi,nbus))
    for i in range(npi):
        m = int(fbus[ppi[i]])
        for k in range(nbus):
            if k == m-1:
                for n in range(nbus):
                    H22[i,k] = H22[i,k] + V[n]*(G[m-1,n]*math.cos(dell[m-1]-dell[n]) + B[m-1,n]*math.sin(dell[m-1]-dell[n]))
                
                H22[i,k] = H22[i,k] + V[m-1]*G[m-1,m-1]
            else:
                H22[i,k] = V[m-1]*(G[m-1,k]*math.cos(dell[m-1]-dell[k]) + B[m-1,k]*math.sin(dell[m-1]-dell[k]))



    # H31 - Derivative of Reactive Power Injections with Angles..
    H31 = np.zeros((nqi,nbus-1))
    for i in range(nqi):
        m = int(fbus[qi[i]])
        for k in range(nbus-1):
            if k+1 == m-1:
                for n in range(nbus):
                    H31[i,k] = H31[i,k] + V[m-1]* V[n]*(G[m-1,n]*math.cos(dell[m-1]-dell[n]) + B[m-1,n]*math.sin(dell[m-1]-dell[n]))
                
                H31[i,k] = H31[i,k] - V[m-1]*V[m-1]*G[m-1,m-1]
            else:
                H31[i,k] = V[m-1]* V[k+1]*(-G[m-1,k+1]*math.cos(dell[m-1]-dell[k+1]) - B[m-1,k+1]*math.sin(dell[m-1]-dell[k+1]))



    # H32 - Derivative of Reactive Power Injections with V..
    H32 = np.zeros((nqi,nbus))
    for i in range(nqi):
        m = int(fbus[qi[i]])
        for k in range(nbus):
            if k == m-1:
                for n in range(nbus):
                    H32[i,k] = H32[i,k] +  V[n]*(G[m-1,n]*math.sin(dell[m-1]-dell[n]) + B[m-1,n]*math.cos(dell[m-1]-dell[n]))
                
                H32[i,k] = H32[i,k] - V[m-1]*B[m-1,m-1]
            else:
                H32[i,k] = V[m-1]* (-G[m-1,k]*math.sin(dell[m-1]-dell[k]) - B[m-1,k]*math.cos(dell[m-1]-dell[k]))



    # H41 - Derivative of Real Power Flows with Angles..
    H41 = np.zeros((npf,nbus-1))
    for i in range(npf):
        m = int(fbus[pf[i]])
        n = int(tbus[pf[i]])
        for k in range(nbus-1):
            if k+1 == m-1:
                H41[i,k] = V[m-1]*V[n-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]) + B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]))
            elif k+1 == n-1:
                H41[i,k] = -V[m-1]*V[n-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]) + B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]))
            else:
                H41[i,k] =0


    # H42 - Derivative of Real Power Flows with V..
    H42 = np.zeros((npf,nbus))
    for i in range(npf):
        m = int(fbus[pf[i]])
        n = int(tbus[pf[i]])
        for k in range(nbus):
            if k == m-1:
                H42[i,k] = -V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]) - B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))-2*G[m-1,n-1]*V[m-1]
            elif k == n-1:
                H42[i,k] = -V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]) - B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))
            else:
                H42[i,k] =0


    # H51 - Derivative of Reactive Power Flows with Angles..
    H51 = np.zeros((nqf,nbus-1))
    for i in range(nqf):
        m = int(fbus[qf[i]])
        n = int(tbus[qf[i]])
        for k in range(nbus-1):
            if k+1 == m-1:
                H51[i,k] = -V[m-1]*V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]) - B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))
            elif k+1 == n-1:
                H51[i,k] = V[m-1]*V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]) - B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))
            else:
                H51[i,k] =0


    # H52 - Derivative of Reactive Power Flows with V..
    H52 = np.zeros((nqf,nbus))
    for i in range(nqf):
        m = int(fbus[qf[i]])
        n = int(tbus[qf[i]])
        for k in range(nbus):
            if k == m-1:
                H52[i,k] = -V[n-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]) + B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1])) - 2*V[m-1]*(-B[m-1,n-1]+ bpq[m-1,n-1])
            elif k == n-1:
                H52[i,k] = -V[m-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]) + B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]))
            else:
                H52[i,k] =0


    # Measurement Jacobian, H..
    H=[]
    for i in range(H11.shape[0]):
        row=[]
        for k in range(H11.shape[1]):
            row.append(H11[i,k])
        for n in range(H12.shape[1]):
            row.append(H12[i,n])    
        H.append(row)
    
    for i in range(H21.shape[0]):
        row=[]
        for k in range(H21.shape[1]):
            row.append(H21[i,k])
        for n in range(H22.shape[1]):
            row.append(H22[i,n])    
        H.append(row)
    for i in range(H31.shape[0]):
        row=[]
        for k in range(H31.shape[1]):
            row.append(H31[i,k])
        for n in range(H32.shape[1]):
            row.append(H32[i,n])    
        H.append(row)
    for i in range(H41.shape[0]):
        row=[]
        for k in range(H41.shape[1]):
            row.append(H41[i,k])
        for n in range(H42.shape[1]):
            row.append(H42[i,n])    
        H.append(row)
    for i in range(H51.shape[0]):
        row=[]
        for k in range(H51.shape[1]):
            row.append(H51[i,k])
        for n in range(H52.shape[1]):
            row.append(H52[i,n])    
        H.append(row)

    
    # transposed H
    Ht=np.transpose(H)
    # inv of Ri
    R=np.linalg.inv(Ri)
    
    yu=np.dot(Ht,R)
    
    H=np.asarray(H)

    # Gain Matrix, Gm..
    result=np.zeros((11,11))
    Gm=yu.dot(H)
    for i in range(len(yu)):
    #iterate through columns of Y
        for j in range(len(H[0])):
             # iterate through rows of Y
            for k in range(len(H)):
                 result[i][j] += round(yu[i][k],3) * round(H[k][j],3)

    print('result')
    print(np.around(result, decimals=3))
    
      
    r2=np.zeros((len(r)))

    for i in range(len(r)):
        r2[i]=r[i]*r[i]
    
    # Objective Function..
    J=sum(np.dot(R,r2))
    print('J:') 
    print(J)



    Gmi=np.linalg.inv(Gm)
    # State Vector 
    fi= np.dot(np.dot(Ht,R),r)
    dE=Gmi.dot(fi)
    print('Gmi:')
    print(Gmi)

 

    iter = iter + 1

    for i in range(len(E)):
        E[i]=E[i]+dE[i]
    

    dell[1:]=E[0:nbus-1]

    V=E[nbus-1:]
    
    V=np.asarray(V)
    print('V:')
    print(V)
    print('dell')
    print(dell)
    print(iter)
    #tol = max(abs(dE))


Dell=(180/math.pi)*dell
print(Dell)
print(V)

