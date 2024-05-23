import numpy as np
def activfun_sigmoid(x, deriv=False):
   if deriv:
      return x * (1 - x)
   return 1 / (1 + np.exp( -x))

X = np.array([[1,1,0,1], [0,0,0,0], [0,1,1,1], [1,1,1,1], [0,1,0,1], [0,1,0,0], [1,0,0,1], [0,1,0,0]])


Y = np.array([[0, 0, 1, 0, 1, 0, 1, 0]]).T

np.random.seed(1)  


w0 = 2*np.random.random((4, 1)) -1

print("Initial weights -\n", w0)

for n in range(1200):
   l_input = X
  
   l_output = activfun_sigmoid(l_input.dot(w0)) 

   
   l_output_error = Y - l_output 
   l_output_delta = l_output_error * activfun_sigmoid(l_output, True) 

   
   w0 = w0 + 2*l_input.T.dot(l_output_delta)


print("\nFinal weights -\n",w0)
print("\nOutout After Training:\n",l_output)
print("\nLoss: \n"+str(np.mean(np.square(Y - l_output))))


"""OUTPUT:- 
   Initial weights -
   [[-0.16595599]
   [ 0.44064899]
   [-0.99977125]
   [-0.39533485]]

   Final weights -
   [[-7.25597252]
   [-6.79764545]
   [-0.17634418]
   [10.60145177]]

   Outout After Training:
   [[0.0307177 ]
   [0.5       ]
   [0.97409355]
   [0.02588   ]
   [0.97819058]
   [0.00111616]
   [0.96594163]
   [0.00111616]]

   Loss:
   0.03174032635521366
"""
