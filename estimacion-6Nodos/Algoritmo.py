import numpy as np
import math
import sys


num=6 # numero de buses 
#         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
#         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |

linedata6 =[[1, 2, complex(0.1, 0.2),    complex(0, 0.02), 1],
            [1, 4, complex(0.05, 0.2),   complex(0, 0.02), 1],
            [1, 5, complex(0.08, 0.3),   complex(0, 0.03), 1],
            [2, 3, complex(0.05, 0.025), complex(0, 0.03), 1],
            [2, 4, complex(0.05, 0.01),  complex(0, 0.05), 1],
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
           [3,    2 ,   0.50 ,     2,      0 ,  1e-4],
           [6,    2 ,   -0.70 ,    4,      0 ,  1e-4],
           [7,    2,    -0.70 ,    5,      0 ,  1e-4],

           #------------------------------------%
           #---- Reative Power Injection -------%
           [8,     3,   0.7436,     2,      0 ,  1e-4],
           [10,    3,   -0.70 ,     4,      0 ,  1e-4],
           [11,    3,   -0.70 ,     5,      0 ,  1e-4],

           #------------------------------------%
           #------ Real Power Flow ------------- %
           [18,     4,   -0.2778,   2,       1,  64e-6],
           [21,     4,    0.0293,   2,       3,  64e-6],
           [22,     4,    0.3309,   2,       4,  64e-6],
           [23,     4,    0.1551,   2,       5,  64e-6],
           [24,     4,    0.2625,   2,       6,  64e-6],
           [25,     4,    0.0408,   4,       5,  64e-6],
           [26,     4,   -0.1802,   5,       3,  64e-6],
           [28,     4,    0.0161,   5,       6,  64e-6],
           #----   --------------------------------%
           #------ Real Power Flow ------------- %
           [18,     5,    0.1282,   2,       1,  64e-6],
           [21,     5,   -0.1227,   2,       3,  64e-6],
           [22,     5,    0.4605,   2,       4,  64e-6],
           [23,     5,    0.1535,   2,       5,  64e-6],
           [24,     5,    0.1240,   2,       6,  64e-6],
           [25,     5,   -0.0494,   4,       5,  64e-6],
           [26,     5,   -0.2610,   5,       3,  64e-6],
           [28,     5,   -0.0966,   5,       6,  64e-6],]
           #--------------------------------------%


linedata14 =[[1, 2, complex(0.01938,   0.05917), complex(0, 0.0264),         1],
            [1,      5,  complex(0.05403,   0.22304), complex(0, 0.0246) ,        1],
            [2,      3,  complex(0.04699,   0.19797), complex(0, 0.0219) ,        1],
            [2,      4,  complex(0.05811,   0.17632), complex(0, 0.0170) ,        1],
            [2,      5,  complex(0.05695,   0.17388), complex(0, 0.0173) ,        1],
            [3,      4,  complex(0.06701,   0.17103), complex(0, 0.0064) ,        1],
            [4,      5,  complex(0.01335,   0.04211), complex(0, 0.0)    ,        1],
            [4,      7,  complex(0.0,       0.20912), complex(0, 0.0)     ,   0.978],
            [4,      9,  complex(0.0,       0.55618), complex(0, 0.0)     ,   0.969],
            [5,      6,  complex(0.0,       0.25202), complex(0, 0.0)    ,   0.932],
            [6,     11,  complex(0.09498,   0.19890), complex(0, 0.0)    ,        1],
            [6,     12,  complex(0.12291,   0.25581), complex(0, 0.0)    ,        1],
            [6,     13,  complex(0.06615,   0.13027), complex(0, 0.0)    ,        1],
            [7,      8,  complex(0.0,       0.17615), complex(0, 0.0)    ,       1],
            [7,      9,  complex(0.0,      0.11001 ), complex(0, 0.0)     ,       1],
            [9,     10,  complex(0.03181,   0.08450), complex(0, 0.0)    ,        1],
            [9,     14,  complex(0.12711,   0.27038), complex(0, 0.0)    ,        1],
            [10,     11, complex(0.08205,   0.19207), complex(0, 0.0)   ,         1],
            [12,     13, complex(0.22092,   0.19988), complex(0, 0.0)   ,         1],
            [13,     14, complex(0.17093,   0.34802), complex(0, 0.0)   ,         1 ]]
                 

#         |Msnt |Type | Value | From | To | Rii | 
zdata14   = [ #---- Voltage Magnitude ------------%
            [1,     1,    1.06,    1,   0,  9e-4],
            #-----------------------------------%
            #---- Real Power Injection ---------%
           [2,     2,    0.1830,  2,       0 ,  1e-4],
           [3,    2 ,  -0.9420 ,  3 ,      0 ,  1e-4],
           [4,     2,    0.00  ,  7 ,      0 ,  1e-4],
           [5,     2,    0.00  ,   8,       0,   1e-4], 
           [6,     2,   -0.0900,  10,       0,   1e-4],
           [7,     2,   -0.0350,  11,       0,   1e-4],
           [8,     2,   -0.0610,  12,       0,   1e-4],
           [9,     2,   -0.1490,  14,       0,   1e-4],
           #------------------------------------%
           #---- Reative Power Injection -------%
           [10,     3,    0.3523,   2,       0 ,  1e-4],
           [11,     3,   0.0876,   3 ,      0 ,  1e-4],
           [12,     3,    0.00  ,  7 ,      0 ,  1e-4],
           [13,     3,    0.2103,   8,       0,   1e-4], 
           [14,     3,   -0.0580,  10,       0,   1e-4],
           [15,     3,   -0.0180,  11,       0,   1e-4],
           [16,     3,   -0.0160,  12,       0,   1e-4],
           [17,     3,   -0.0500,  14,       0,   1e-4],
           #------------------------------------%
           #------ Real Power Flow ------------- %
           [18,     4,    1.5708,   1,       2,   64e-6],
           [19,     4,    0.7340,   2,       3,   64e-6],
           [20,     4,   -0.5427,   4,       2,   64e-6],
           [21,     4,    0.2707,   4,       7,   64e-6],
           [22,     4,    0.1546,   4,       9,  64e-6],
           [23,     4,   -0.4081,   5,       2,   64e-6],
           [24,     4,    0.6006,   5,       4,   64e-6],
           [25,     4,    0.4589,  5 ,      6 ,  64e-6],
           [26,     4,    0.1834,   6,      13,   64e-6],
           [27,     4,    0.2707,   7,       9,   64e-6],
           [28,     4,   -0.0816,  11,      6 ,  64e-6],
           [29,     4,    0.0188,  12,      13,   64e-6],
           #------------------------------------%
           #------ Real Power Flow ------------- %
           [30,     5,   -0.1748,  1,       2,   64e-6],
           [31,     5,    0.0594,   2,       3,  64e-6],
           [32,     5,    0.0213,   4,       2,   64e-6],
           [33,     5,   -0.1540,   4,       7,   64e-6],
           [34,     5,   -0.0264,   4,       9,   64e-6],
           [35,     5,   -0.0193,   5,       2,   64e-6],
           [36,     5,   -0.1006,  5 ,      4 ,  64e-6],
           [37,     5,   -0.2084,   5,       6,   64e-6],
           [38,     5,    0.0998,   6,      13,   64e-6],
           [39,     5,    0.1480,   7,       9,   64e-6],
           [40,     5,   -0.0864,  11,       6,   64e-6],
           [41,     5,    0.0141,  12,      13,   64e-6]]
           #--------------------------------------%

def ybusppg(num):
    fb = np.array(linedata14)[:,0]      # From bus number...
    tb = np.array(linedata14)[:,1]      # To bus number...
    z = np.array(linedata14)[:,2]       # Z matrix....
    b = np.array(linedata14)[:,3]       # Ground Admittance, B/2...
    a = np.array(linedata14)[:,4]       # Tap setting value..
    y=1/z                               # To get inverse of each element...  
                                        
    nbus = 14                           # number of buses
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

    fb = np.array(linedata14)[:,0]          # From bus number...
    tb = np.array(linedata14)[:,1]          # To bus number...
    b = np.array(linedata14)[:,3]           # Ground Admittance, B/2...
    nbus = int(max(max(fb), max(tb)))       # no. of buses...
    nbrach = len(fb)                        # no. of branches...
    
    bbus = np.zeros((nbus,nbus),dtype = 'complex_')
    for k in range(nbrach):
        bbus[int(fb[k])-1,int(tb[k])-1] = b[k]
        bbus[int(tb[k])-1,int(fb[k])-1] = bbus[int(fb[k])-1,int(tb[k])-1]
    return bbus.imag


#% IEEE - 14 or IEEE - 30 bus system..(for IEEE-14 bus system replace 30 by 14)...
ybus = ybusppg(num)                     # Get YBus..
bpq = bbusppg(num)                      # Get B data..
nbus = 14                               # Get number of buses..
typee = np.array(zdata14)[:,1]          # Type of measurement, Vi - 1, Pi - 2, Qi - 3, Pij - 4, Qij - 5, Iij - 6..
z = np.array(zdata14)[:,2]              # Measuement values..
fbus = np.array(zdata14)[:,3]           # From bus..
tbus = np.array(zdata14)[:,4]           # To bus..
Ri = np.diag(np.array(zdata14)[:,5])    # % Measurement Error..
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
tol=5

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
          
        
    for i in range(nqi):
        m = int(fbus[qi[i]])
        for k in range(nbus):
            h3[i] = h3[i] + V[m-1]*V[k]*(G[m-1,k]*math.sin(dell[m-1]-dell[k]) - B[m-1,k]*math.cos(dell[m-1]-dell[k]))
        h.append(h3[i])
    for i in range(npf):
        m = int(fbus[pf[i]])
        n = int(tbus[pf[i]])
        a= V[m-1]*V[m-1]*G[m-1,n-1]
        b= V[m-1]*V[n-1]*(-G[m-1,n-1]*math.cos(dell[m-1]-dell[n-1])- B[m-1,n-1]*math.sin(dell[m-1]-dell[n-1]))
        h4[i] =-a-b
        h.append(h4[i])

    for i in range(nqf):
        m = int(fbus[qf[i]])
        n = int(tbus[qf[i]])
        a=V[m-1]*V[m-1]*(-B[m-1,n-1]+bpq[m-1,n-1])
        b=V[m-1]*V[n-1]*(-G[m-1,n-1]*math.sin(dell[m-1]-dell[n-1])+ B[m-1,n-1]*math.cos(dell[m-1]-dell[n-1]))
        h5[i] = -a-b
        h.append(h5[i])

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

    

    # Gain Matrix, Gm..
    Gm=np.dot(Ht,np.dot(R,H))
       
    r2=np.zeros((len(r)))
        
    for i in range(len(r)):
        r2[i]=r[i]*r[i]
    
    # Objective Function..
    J=sum(np.dot(R,r2))
    
    Gmi=np.linalg.inv(Gm)
    # State Vector    
    dE=np.dot(Gmi,np.dot(np.dot(Ht,R),r))

    iter = iter + 1

    for i in range(len(E)):
        E[i]=E[i]+dE[i]
    

    dell[1:]=E[0:nbus-1]
    V=E[nbus-1:]
    V=np.asarray(V)
  
    tol = max(abs(dE))

    print('tolerancia')
    print(tol)

    print(iter)

Dell=(180/math.pi)*dell
print(Dell)
print(V)
