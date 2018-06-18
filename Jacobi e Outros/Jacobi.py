
# coding: utf-8

# ## Alunos:
# ## Matrícula:
# 

# In[1]:


import numpy as np
import scipy
import math


# # Método Iterativo de Jacobi:
# 
# ## $$ x^{(1)} = Aproximação inicial $$
# 
# ## $$x_{i}^{(k+1)} = \frac{(y_{i} - \Sigma_{j=1}^{n} a_{i,j}x_{j}^{(k)})}{a_{ii}}$$
# 

# # Convergência: Critério das Linhas
# 
# ## $$\Sigma_{j=1, j \neq i}^{n} \mid a_{i,j} \mid <  \mid a_{i,i} \mid $$

# In[2]:


def criterio_linhas(A):
    swap = np.copy(A)
    b = np.diag(A)
    A = A - np.diagflat(b)
    x = np.ones(b.size)
    permutation = b.size**2
    acc = True
    
    while(acc and permutation > 0):
        for i in range(b.size):
            x[i] = np.sum(A[i])/b[i]
        
        if(np.amax(x) < 1): acc = False
        else:
            permutation = permutation-1
            swap = np.random.permutation(swap)
            A = np.copy(swap)
            b = np.diag(A)
            A = A - np.diagflat(b)
    
    return np.amax(x)


# In[3]:


#Matriz A, Matriz b dos termos independentes e N o número de iterações e o erro
def jacobi(A, b, N, chute, erro = 0.00000001):
    if(criterio_linhas(A) > 1):
        print("O sistema não converge para o método de Jacobi")
        return
    
    x = np.diag(A) #recebe um vetor contendo a diagonal principal
    A = A - np.diagflat(x) #Zera a diagonal principal de A
    
    #Para dividir todos os valores da matriz A pelos termos independentes
    for i in range(x.size):
        A[i] = A[i]/x[i]
        b[i] = b[i]/x[i]

    x = np.copy(chute)
    swap = np.zeros(x.size)
    
    A = A*-1
    
    for stop in range(N):
        for i in range(x.size):
            swap[i] = np.sum((A[i]*x))+(b[i])
        #Cálculo da tolerância ou erro
        print(f"Iteração {stop}: {swap}")
        if((np.linalg.norm(swap) - np.linalg.norm(x)) < erro): return swap
        x = np.copy(swap)

    return x


# In[4]:


mat = np.array([[10.0,2.0,1.0],[1.0,5.0,1.0],[2.0,3.0,10.0]])
ind = np.array([7.0,-8.0,6.0])
chute1 = np.zeros(3)

mat2 = np.array([[-3.0,1.0,1.0],[2.0,5.0,1.0],[2.0,3.0,7.0]])
ind2 = np.array([2.0,5.0,-17.0])
chute2 = [1.0, 1.0, -1.0]


# In[5]:


jacobi(mat, ind, 3, chute1)


# In[6]:


jacobi(mat2, ind2, 4, chute2)


# # Produto Interno entre dois vetores:
# 
# ## $$< u  , v > = a.a' + b.b'$$

# In[7]:


def produto_interno(u, v):
    resultado = 0
    
    if(u.size != v.size):
        print("Vetores com tamanhos diferentes")
        return
    
    for i in range(u.size):
           resultado += u[i]*v[i]

    return resultado


# In[33]:


v = np.array([1,-1,1])
u = np.array([1,1,0])
v1 = np.array([1, -2, 2])
u1 = np.array([-3, 6, -6])


# In[35]:


produto_interno(u, v)
produto_interno(u1, v1)


# # Ângulo Entre Vetores:
# 
# ##  $$cos (\theta) = \frac{< a  , b >}{\mid a \mid \mid b \mid}$$

# In[45]:


def angulo_vetores(u, v):
    if(u.size != v.size):
        print("Vetores com tamanhos diferentes")
        return
    
    prodint = produto_interno(u, v)
    n1 = np.linalg.norm(u)
    n2 = np.linalg.norm(v)
    total = prodint/(n1*n2)
    totalcos = np.cos(total)
    totalarccos = np.arccos(totalcos)
    print(f"total radians: {total} cos: {totalcos} arccos: {totalarccos}")
    
    return (total, totalcos, totalarccos)


# In[46]:


angulo_vetores(u, v)
angulo_vetores(u1, v1)

