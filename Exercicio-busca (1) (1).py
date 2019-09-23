#!/usr/bin/env python
# coding: utf-8

# # Inteligência Artificial

# In[7]:


from pprint import pprint
from copy import deepcopy

# 1. Estado inicial
estado_inicial = {
    'A': [5,4,3, 2, 1],
    'B': [],         
    'C': [],
}

# 2. Ações
acoes = [
    'mover_A_B',
    'mover_A_C',
    'mover_B_C',
    'mover_B_A',
    'mover_C_A',
    'mover_C_B',
]

def mover(estado, origem, destino):
    if not estado[origem]:
        return None
    if not estado[destino]:
        estado = deepcopy(estado)
        estado[destino].append(estado[origem].pop())
        return estado
    elif estado[destino][-1] > estado[origem][-1]:
        estado = deepcopy(estado)
        estado[destino].append(estado[origem].pop())
        return estado
    return None

# 3. Transição de estados
# Dada um estado e uma acao, o modelo de transicao de estados
# retorna todos os estados vizinhos (fronteira)
def sucessores(estado):
    
    # Gera todos os estados sucessores
    fronteira = [
        mover(estado, 'A', 'B'),
        mover(estado, 'A', 'C'),
        mover(estado, 'B', 'A'),
        mover(estado, 'B', 'C'),
        mover(estado, 'C', 'A'),
        mover(estado, 'C', 'B'),
    ]
    
    return [estado for estado in fronteira if estado]

# 4. Estado objetivo
estado_final = {
    'A': [],
    'B': [],         
    'C': [5,4,3, 2, 1],
}

# 5. Custo da solução
# Custo eh calculado como o numero de operacoes
def solucao(estado):
    return estado


# # Busca em largura

# In[60]:


estados_passados = []


def busca_em_largura(fronteira):
    if not fronteira:
        return None
    
    nodo = fronteira.pop(0)
    
    if nodo == estado_final:
        print('--------------------------------------------------------------------------\n')
        return solucao(nodo)
    
    if nodo not in estados_passados:
        estados_passados.append(nodo)
        vizinhos = sucessores(nodo)        
        fronteira += vizinhos
    return nodo 
        


# In[61]:


fronteira = [estado_inicial]

while fronteira:
    result=busca_em_largura(fronteira)
    pprint(result)
    if(result==estado_final):
        break


# 

# In[11]:





# # Busca em profundidade

# In[1]:





from copy import deepcopy


class pilha: 
    	def __init__(self):
    		self.p = []
    		
    	def push(self,t):
    		if(self.pushable(t)):
    			self.p.append(t)
    			return True
    		else:
    			return False
    			
    	def pop(self):
    		return self.p.pop()
    		
    	def top(self):
    		if self.p:
    			return self.p[-1]
    		
    	def remove(self):
    		self.p.pop(0)
    		
    	def isEmpty(self):
    		return (len(self.p)==0)
    		
    	def printStack(self):
    		print (self.p)
    		
    	def isFull(self):
    		return (len(self.p) >= 5)
    		
    	def pushable(self,t):
    		if (self.isFull()):
    			return False
    		else:
    			return self.isEmpty() or t < self.top()
    			
    	def size(self):
    		return len(self.p)
    	def equal(self, s):
    		return self.p == s.p
    		

class torre:
	def __init__(self):
		
		self.t1 = pilha()
		self.t2 = pilha()
		self.t3 = pilha()
		self.stacks = [self.t1,self.t2,self.t3]
		
		
		self.allMoves = ((0,1) , (0,2) , (1,0) , (1,2) ,
									(2,0) , (2,1))
		self.path = []
		self.validMoves = []
		self.lastMove = ()
		self.steps = 0
					
		self.setup()
		
	
	def setup(self):
		for x in range (5,0,-1):
			self.t3.push(x)
		self.generateMoves()
		
	def generateMoves(self):
		moves = list(self.allMoves)
		
		for move in self.allMoves:
			fromStack = self.stacks[move[1]]
			toStack = self.stacks[move[0]]
			
			if fromStack.top():
				
				if not toStack.pushable(fromStack.top()):
				
					moves.remove(move)
			
			else:
				moves.remove(move)
		
			
		if self.lastMove in moves:
			
			moves.remove(self.lastMove)
		self.validMoves = moves
		
	
		
	def move(self,t):
		if t in self.validMoves:
			fromStack = self.stacks[t[1]]
			toStack = self.stacks[t[0]]
			if fromStack.top():
				toStack.push(fromStack.pop())
			self.lastMove = t
			self.generateMoves()
			self.steps += 1
			self.path.append(t)
			
			
	def moveSmallestLeft(self):
		fromstack = None
		tostack = None
		for stack in self.stacks:
			if stack.top() == 1:
				fromstack = self.stacks.index(stack)
				tostack = fromstack - 1
				if tostack == -1:
					tostack = 2
			
		
		self.move((fromstack, tostack))
			
	
	def solved(self):
		return self.stacks[0].size() == 5
		
	def toString(self):
		for pilha in self.stacks:
			pilha.printStack()
			
	def equal(self, t):
		return self.t1.equal(t.t1) and self.t2.equal(t.t2) and self.t3.equal(t.t3)



def crianca(children, open_states, closed):
	for child in children[:]:
		if open_states:
			for state in open_states:
				if child.equal(state):
					children.remove(child)
		if closed:
			for state in closed:
				if child.equal(state):
					children.remove(child)
	return children
	

def buscaprofundidade():
	start = torre()
	print ("Estado Inicial:")
	start.toString()
	print ("")
	open_states = [start]
	closed = list()
	while open_states:
		X = open_states.pop()
		if X.solved():
			return X
		else:
			X.generateMoves
			moves = X.validMoves
			
			children = []
			for move in moves:
				Y = deepcopy(X)
				Y.move(move)
				children.append(Y)
			
			closed.append(X)
			
			children = crianca(children, open_states, closed)
			
			open_states.extend(children)
		
def main():
	solved = buscaprofundidade()
	print ("Estado Final:")
	solved.toString()
	print (solved.path)

   

if __name__ == '__main__':
	main()


# # Busca A*

# In[2]:




class Torre(object):

    def __init__(self, rings=[], max_rings=0):
        if len(rings) > 0:
            self.rings = rings
        else:
            self.rings = [i for i in range(max_rings, 0, -1)]

    def __iter__(self):
        return iter(self.rings)

 

class jogo(object):
    def __init__(self, A, C , B, parent, end_goal=False):
        self.state = ([i for i in A],
                      [i for i in B],
                      [i for i in C])
        self.end_goal = end_goal
        self.value = None
        self.parent = parent

    def __eq__(self, other):
        if isinstance(other, jogo):
            try:
                for t_index in range(0, len(self.state)):
                    for r_index in range(0, len(self.state[t_index])):
                        if self.state[t_index][r_index] != other.state[t_index][r_index]:
                            return False
                for t_index in range(0, len(other.state)):
                    for r_index in range(0, len(other.state[t_index])):
                        if other.state[t_index][r_index] != self.state[t_index][r_index]:
                            return False
            except IndexError:
                return False
            return True

    def E(self, cost, final_state):
        self.value = self.G(cost) + self.H(final_state)
        return self.value

    def R(self, cost):
        
        return cost + 1

    def T(self, final_state):
      
        if not isinstance(final_state, jogo):
            raise Exception("Erro")

        torreatual_1 = self.state[0]
        torreatual_2 = self.state[1]
        torreatual_3 = self.state[2]
        torrefinal_1 = final_state.state[0]
        torrefinal_2 = final_state.state[1]
        torrefinal_3 = final_state.state[2]

        v = 0

        try:
            for r_index in range(0, len(torreatual_1)):
                if torreatual_1[r_index] == torrefinal_1[r_index]:
                    v -= 1
        except IndexError:
            pass

        try:
            for r_index in range(0, len(torreatual_2)):
                if torreatual_2[r_index] == torrefinal_2[r_index]:
                    v -= 1
        except IndexError:
            pass

        try:
            for r_index in range(0, len(torreatual_3)):
                if torreatual_3[r_index] == torrefinal_3[r_index]:
                    v -= 1
        except IndexError:
            pass

        return v

class Hanoi(object):

    def __init__(self, initial_state, final_state):
        self.final_state = final_state
        self.initial_state = initial_state
        self.open_list = []
        self.closed_list = []

   
    def rh(cls, disc, src, aux, dst):
        if disc > 0:
            cls.recursive_hanoi(disc-1, src, dst, aux)
            cls.recursive_hanoi(disc-1, aux, src, dst)

    def p(self, cost):
        minor_F = 1000000
        index = 0
        index_minor = 1000000
        for node in self.open_list:
            node_F = node.F(cost=cost, final_state=self.final_state)
            if node_F < minor_F:
                minor_F = node_F
                index_minor = index
            index += 1

        selected = self.open_list[index_minor]
        del self.open_list[index_minor]
        return selected


T1 = Torre(max_rings=5)
T2 = Torre()
T3 = Torre()
TA = Torre()
TB = Torre(max_rings=5)
TC = Torre()

I = jogo(T1, T2, T3, parent=None)
F = jogo(TA, TB, TC, parent=None, end_goal=True)

print ("Estado Inicial A*:") 
print (str(I.state))
print ("Estado Final A*:") 
print (str(F.state))




# In[ ]:




