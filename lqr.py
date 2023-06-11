import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def transp(A):
    if (isinstance(A,np.ndarray)):
        return A.transpose()
    else:
        return A
    
def mul(A,B):
    if (isinstance(A,np.ndarray) and isinstance(B,np.ndarray)):
        return A.dot(B)
    else:
        return A*B
    
def inverse(A):
    if (isinstance(A,np.ndarray)):
        return np.linalg.inv(A)
    else:
        return 1/A
        

def LQR(A,B,C,k,N,x_0,r,P,Q,R):
    sum=0
    end=len(r)
    S = np.zeros(N+1)
    v = np.zeros(N+1)
    
    temp=mul(mul(transp(C), P), r[end-1])
    temp=temp.subs(k,N)
    v[N] = temp
    
    temp=mul(mul(transp(C), P), C)
    temp=temp.subs(k,N)
    S[N] = temp
    
    x = np.zeros(N)
    Y = np.zeros(N)
    U = np.zeros(N)
    x[0] = x_0
    if (isinstance(B,np.ndarray)):
        I = np.eye(len(B))
    else:
        I=1
        
    for i in range(N-1,0,-1):        
        temp=mul(mul(mul(transp(A),S[i+1]),(inverse(I+mul(mul(mul(B,inverse(R)),transp(B)),S[i+1])))),A) + mul(mul(transp(C),Q),C)
        temp=temp.subs(k,i)
        S[i] = temp
        
        temp=-mul(mul(mul(mul(mul(mul(transp(A),S[i+1]), inverse(I+mul(mul(mul(B,inverse(R)),transp(B)),S[i+1])) ),B),inverse(R)),transp(B)),v[i+1])+mul(transp(A),v[i+1])+mul(mul(transp(C),Q),r[i])
        temp=temp.subs(k,i)
        v[i] = temp        
   
    
    for i in range(0,N-1,1):
        temp=mul(mul(mul(inverse(I+mul(mul(mul(inverse(R),transp(B)),S[i+1]),B)),inverse(R)),transp(B)), v[i+1]-mul(mul(S[i+1],A),x[i]))
        temp=temp.subs(k,i)
        U[i] = temp

        temp=mul(A,x[i])+mul(B,U[i])
        temp=temp.subs(k,i)
        x[i+1] = temp

        temp=mul(C,x[i])
        temp=temp.subs(k,i)
        Y[i] = temp
    
    temp=mul(C,x[N-1])
    temp=temp.subs(k,N)
    Y[N-1]=temp
    
    sum_partial=0
    for i in range(0,N-1,1):
        sum_partial=mul(mul(transp(mul(C,x[i])-r[i]),Q),(mul(C,x[i])-r[i]))+mul(mul(transp(U[i]),R),U[i])
        sum_partial=sum_partial.subs(k,i)
        sum+=sum_partial
    
    J=0.5*mul(mul(transp(mul(C,x[N-1])-r[end-1]),P),(mul(C,x[N-1])-r[end-1]))+0.5*sum
    J=J.subs(k,i)

    t=np.array(range(0,len(r),1))
    plt.plot(t,x)
    plt.title("States")
    plt.show()
    
    return U,J,Y,S,v


r3=np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13 ,12, 11, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 20, 20, 20, 20, 20, 20, 15, 15, 15, 15])

P=2
Q=2
R=100

k=sym.var('k')
A=k
B=k+1
C=2+k

[U, J, Y, S, v] = LQR(A,B,C,k,len(r3),0,r3,P,Q,R)
print(J)
t=np.array(range(0,len(r3),1))
plt.plot(t,r3)
plt.plot(t,Y)
plt.title("Trajectory tracking")
plt.show()

plt.plot(t,r3-Y)
plt.title("Error diagram")
plt.show()

plt.plot(t,U)
plt.title("Control diagram")
plt.show()

t2=np.array(range(0,len(r3)+1,1))

plt.plot(t2,S)
plt.title("S")
plt.show()

plt.plot(t2,v)
plt.title("v")
plt.show()
