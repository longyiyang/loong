import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data
        self.gard=None
        self.creator=None
    
    def set_creator(self,func):
        self.creator=func

    def backward(self):
        funcs= [self.creator]
        while funcs:
            if self.grad is None:
                self.grad = np.ones_like(self.data)
            f=funcs.pop()
            x,y=f.input,f.output
            x.grad=f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:

    def __call__(self,input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        output.set_creator(self)
        self.input=input
        self.output=output  #将当前Function这个实例设置为变量output的“创建者”
        return output           
    
    def forward(self,x):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError
    
class Square(Function):
    
    def forward(self,x):
        y=x ** 2
        return y
      
    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y
    
    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx
    
#库函数实现

def square(x):
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)

x=Variable(np.array(0.5))
y=square(exp(square(x)))
y.grad=np.array(1.0)
y.backward()
print(x.grad)