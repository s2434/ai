# pip install numpy

import numpy as np
def activfun_sigmoid(x,deriv=False): 
   if(deriv==True):
      return x*(1-x)    
   return 1/(1+np.exp(-x))  
   
   
X = np.array([  [1,1,0,1], [0,0,0,0], [0,1,1,1], [1,1,1,1] , [0,1,0,1], [0,1,0,0], [1,0,0,1], [0,1,0,0] ])


y = np.array([[0,0,1,0,1,0,1,0]]).T

np.random.seed(1) 

w0 = 2*np.random.random((4,4)) - 1
w1 = 2*np.random.random((4,1)) - 1
for n in range(140):
   l_input = X  
   
 
   l1 = activfun_sigmoid(l_input.dot(w0))
   l_output = activfun_sigmoid(l1.dot(w1))
   

   l_output_error = y - l_output 
   l_output_delta = l_output_error * activfun_sigmoid(l_output,True)   
   l1_error = l_output_delta.dot(w1.T)
   l1_delta = l1_error * activfun_sigmoid(l1,deriv=True)

   
   w1 = w1 + 2*l1.T.dot(l_output_delta)
   w0 = w0 + 2*l_input.T.dot(l1_delta)

print( "Output After Training:")
print (l_output)
print ("Loss: \n" + str(np.mean(np.square(y - l_output))))


"""OUTPUT:- 
   Output After Training:
   [[0.05450366]        
   [0.07430379]        
   [0.95094455]        
   [0.04364975]        
   [0.95735073]        
   [0.00096452]        
   [0.93477325]        
   [0.00096452]]       
   Loss: 
   0.0023598488208046264
"""
