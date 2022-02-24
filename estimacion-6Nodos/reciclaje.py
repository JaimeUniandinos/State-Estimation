'''
X12=0.2
X13=0.4
X23=0.25

tetha3=0

M13=0.05 #pu
M32=0.40 #pu
#Se sabe que tetha3=0, y con esto se puede resolver f13 y encontrar tetha1 y f32 encontrar theta2.

tetha1=M13*X13+tetha3
print(tetha1)
tetha2=-M32*X23+tetha3
print(tetha2)
'''
#Xest=[tetha1, tetha2]
'''
f12=1/0.2(tetha1-tetha2)=5tetha1-5tetha2
f12=1/0.4(tetha1-tetha3)=2.5tetha1
f12=1/0.25(tetha3-tetha2)=-4tetha2
'''
'''
Zmeas=[0.62,0.06,0.37]
tetha3=0
H=np.array([[5, -5],[2.5, 0],[0, -4]])
R=np.array([[0.0001, 0, 0],[0, 0.0001, 0],[0, 0, 0.0001]])
Ht=np.transpose(H)
Ri=np.linalg.inv(R)
#Xest=[[H]' [R^-1][H]]^-1 [H]'[R^-1]Zmeas
n=[[1,0,0],[0,0,0]]
p=[[1,0],[0,0],[0,0]]
Xest=np.dot(np.linalg.inv(np.dot(np.dot(Ht,Ri),H)),np.dot(np.dot(Ht,Ri),Zmeas))
print(Xest)
tetha1=Xest[0]lui
tetha2=Xest[1]

f12=5*tetha1-5*tetha2
f13=2.5*tetha1
f32=-4*tetha2

print(f12)
print(f13)
print(f32)
'''

'''
EEEEEEEEEEEEEE
FUNCIONA BIEN DE SOLUCION LINEAL
'''


'''
#Nodos 1 |  2 |  3  |  4|   5|   6
A= [[  0 ,  1 ,  -1 ,  0,   0,   0],#I1
    [  0 ,  1 ,  0 ,  0,   0,   -1],#I2
    [  0 ,  1 ,  0 ,  0,   -1,   0],#I3
    [  -1 ,  1 ,  0 ,  0,   0,   0],#I4
    [  0 ,  1 ,  0 ,  -1,   0,   0],#I5
    [  -1 ,  0 ,  0 ,  1,   0,   0],#I6
    [  0 ,  0 ,  0 ,   1,  -1,   0],#I7
    [  0 ,  -1 ,  0 ,  1,   0,   0],#I8
    [  0 ,  -1 ,  0 ,  0,   1,   0],#I9
    [  0 ,  0 ,  0 ,  0,   1,   -1],#I10
    [  -1 ,  0 ,  0 ,  0,   1,   0],#I11
    [  0 ,  0 ,  -1 ,  0,   1,   0],#I12
    [  0 ,  0 ,  0 ,  -1,   1,   0]]#I13
Y1=ybus[1,2]
Y2=ybus[1,5]
Y3=ybus[1,4]
Y4=ybus[1,0]
Y5=ybus[1,3]
Y6=ybus[3,0]
Y7=ybus[3,4]
Y8=ybus[3,1]
Y9=ybus[4,1]
Y10=ybus[4,5]
Y11=ybus[4,0]
Y12=ybus[4,2]
Y13=ybus[4,3]
Y= [[  Y1 ,  0 ,  0 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  0,   0,   0 ],#I1
    [  0 ,  Y2 ,  0 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,  Y3 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  Y4,   0,   0,  0,  0 , 0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   Y5,   0,  0,  0 , 0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   Y6,  0,  0 , 0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  Y7,  0 , 0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  Y8 , 0 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  0 , Y9 ,   0 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  0 ,  0 , Y10 ,  0,   0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  Y11, 0,   0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  0,   Y12, 0 ],
    [  0 ,  0 ,   0 ,  0,   0,   0,  0,  0 ,  0 ,   0 ,  0,   0,  Y13]]
print('Y2')
print(Y)
YS= [[  0 ,  bpq[1,2] ,  0 ,  0,   0,   0],#I1
    [  0 ,  bpq[1,5] ,  0 ,  0,   0,   0],#I2
    [  0 ,  bpq[1,4] ,  0 ,  0,   0,   0],#I3
    [  0 ,  bpq[1,0] ,  0 ,  0,   0,   0],#I4
    [  0 ,  bpq[1,3] ,  0 ,  0,   0,   0],#I5
    [  0 ,  0 ,  0 ,  bpq[3,0],   0,   0],#I6
    [  0 ,  0 ,  0 ,  bpq[3,4],  0 ,   0],#I7
    [  0 ,  0 ,  0 ,  bpq[3,1],   0,   0],#I8
    [  0 ,  0 ,  0 ,  0,   bpq[4,1],   0],#I9
    [  0 ,  0 ,  0 ,  0,   bpq[4,5],   0],#I10
    [  0 ,  0 ,  0 ,  0,   bpq[4,0],   0],#I11
    [  0 ,  0 ,  0 ,  0,   bpq[4,2],   0],#I12
    [  0 ,  0 ,  0 ,  0,   bpq[4,3],   0]]#I13



E=[VN[1],VN[3],VN[4],
   (VN[2]-VN[1])/(linedata6[3][2]),#I1
   (VN[5]-VN[1])/(linedata6[6][2]),#I2
   (VN[4]-VN[1])/(linedata6[5][2]),#I3
   (VN[0]-VN[1])/(linedata6[0][2]),#I4
   (VN[3]-VN[1])/(linedata6[4][2]),#I5

   (VN[0]-VN[3])/(linedata6[1][2]),#I6
   (VN[4]-VN[3])/(linedata6[9][2]),#I7
   (VN[1]-VN[3])/(linedata6[4][2]),#I8

   (VN[1]-VN[4])/(linedata6[5][2]),#I9
   (VN[5]-VN[4])/(linedata6[10][2]),#I10
   (VN[0]-VN[4])/(linedata6[2][2]),#I11
   (VN[2]-VN[4])/(linedata6[7][2]),#I12
   (VN[3]-VN[4])/(linedata6[9][2])]#I13
E=np.asarray(E)
print(E)

'''