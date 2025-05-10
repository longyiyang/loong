import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
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
        output=Variable(as_array(y))
        output.set_creator(self)
        self.input=input
        self.output=output  #将当前Function这个实例设置为变量output的“创建者”
        return output           
    
    def forward(self,x):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError