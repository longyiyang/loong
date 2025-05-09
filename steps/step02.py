from step01 import Variable
import numpy as np

#相当于父类
class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self,x):
        raise NotImplementedError() #不可直接使用父类，需要子类继承
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    

