import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class ACT_0:
    def __init__(self,inp,**args):
        try:
            self.name = args["name"]
        except:
            self.name = "def_act"
        self.x = inp
        self.bshape = inp.shape[1:]
        self.r = tf.get_variable(self.name+"_r",shape = self.bshape,initializer=tf.contrib.layers.xavier_initializer())
        self.lr = tf.get_variable(self.name+"_lr",shape = self.bshape,initializer=tf.contrib.layers.xavier_initializer())
        self.oup = (tf.nn.tanh(self.x)*tf.nn.sigmoid(self.r)+tf.nn.relu(self.x)*(1-tf.nn.sigmoid(self.r)))*self.lr
    def __call__(self):
        return self.oup
class BN:
    def update(self):
        ema_apply_op = self.ema.apply([self.mean,self.var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(self.mean),tf.identity(self.var)
    def __call__(self):
        return self.oup
    def __init__(self,inp,**args):
        self.nD = len(inp.shape)
        try:
            self.name = args["name"]
        except:
            self.name = "def"
        try:
            self.axes = args["axes"]
        except:
            self.axes = list(range(self.nD-1))
        self.mean,self.var = tf.nn.moments(inp,axes = self.axes)
        self.scale = tf.get_variable(self.name+"_scale",shape=inp.shape[1:],initializer=tf.contrib.layers.xavier_initializer())
        self.shift = tf.get_variable(self.name+"_shift",shape=inp.shape[1:],initializer=tf.contrib.layers.xavier_initializer())
        
        try:
            self.epsilon = args["epsilon"]
        except:
            self.epsilon = 10e-5
        try:
            self.ema = tf.train.ExponentialMovingAverage(decay=args["decay"])
        except:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self.mean,self.var = self.update()
        self.oup = tf.nn.batch_normalization(inp,self.mean,self.var,self.shift,self.scale,self.epsilon)

class CONV_RES:
    def __init__(self,inp,**args):
        try:
            self.name = args["name"]
        except:
            self.name = "defres"
        try:
            self.step = args["step"]
        except:
            self.step = 1
        try:
            f = args["fsize"]
        except:
            f = 3
        try:
            self.bn = args["bn"]
        except:
            self.bn = False
        b,w,h,c = inp.shape
        X = inp
        for i in range(self.step):
            if self.bn:
                X = BN(X,name = self.name+"BN"+str(i))()
            W = tf.get_variable(self.name+"w"+str(i),shape = [f,f,c,c],initializer = tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(self.name+"b"+str(i),shape = [w,h,c],initializer = tf.zeros_initializer())
            Z = tf.nn.conv2d(X,W,strides = [1,1,1,1],padding = "SAME")
            X = tf.nn.relu(Z+b)
        if self.bn:
            X = BN(X,name = self.name + "BNN")()
        W = tf.get_variable(self.name+"W",shape = [f,f,c,c],initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(self.name+"b",shape = [w,h,c],initializer = tf.zeros_initializer())
        self.OUP = tf.nn.relu(tf.nn.conv2d(X,W,strides = [1,1,1,1],padding = "SAME") + inp + b)
    def __call__(self):
        return self.OUP

class GOOG:
    def __init__(self,inp,**args):
        try:
            self.name = args["name"]
        except:
            self.name = "defInc"
        try:
            self.oupc = args["channels"]
        except:
            self.oupc = 1
        b,w,h,c = inp.shape
        w110 = tf.get_variable(self.name+"w110",shape = [1,1,c,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A110 = tf.nn.conv2d(inp,w110,strides = [1,1,1,1],padding = "SAME")
        
        w113 = tf.get_variable(self.name+"w113",shape = [1,1,c,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Z113 = tf.nn.conv2d(inp,w113,strides = [1,1,1,1],padding = "SAME")
        w33 = tf.get_variable(self.name+"w33",shape = [3,3,self.oupc,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A33 = tf.nn.conv2d(Z113,w33,strides = [1,1,1,1],padding = "SAME")
        
        w115 = tf.get_variable(self.name+"w115",shape = [1,1,c,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Z115 = tf.nn.conv2d(inp,w115,strides = [1,1,1,1],padding = "SAME")
        w55 = tf.get_variable(self.name+"w55",shape = [5,5,self.oupc,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A55 = tf.nn.conv2d(Z115,w55,strides = [1,1,1,1],padding = "SAME")
        
        mx = tf.nn.max_pool(inp,ksize = [1,3,3,1],strides = [1,1,1,1],padding="SAME")
        wm = tf.get_variable(self.name + "wm",shape = [1,1,c,self.oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Am = tf.nn.conv2d(mx,wm,strides = [1,1,1,1],padding = "SAME")
        
        self.OUP = tf.concat([A110,A33,A55,Am],axis = 3)
    def __call__(self):
        return self.OUP

class YSW:
    def __init__(self,inp,**args):
        try:
            self.name = args["name"]
        except:
            self.name = "defYSW"
        try:
            oupc = args["channels"]
        except:
            oupc = 1
        b,w,h,c = inp.shape
        w11 = tf.get_variable(self.name + "w11",shape=[1,1,c,oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A11 = tf.nn.conv2d(inp,w11,strides = [1,1,1,1],padding = "SAME")
        
        w33 = tf.get_variable(self.name + "w33",shape = [3,3,c,oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A33 = tf.nn.conv2d(inp,w33,strides = [1,1,1,1],padding = "SAME")
        
        w55 = tf.get_variable(self.name + "w55",shape = [5,5,c,oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        A55 = tf.nn.conv2d(inp,w55,strides = [1,1,1,1],padding = "SAME")
        
        Amx0 = tf.nn.max_pool(inp,ksize=[1,3,3,1],strides=[1,1,1,1],padding="SAME")
        Wmx = tf.get_variable(self.name + "WMX",shape = [1,1,c,oupc],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Amx = tf.nn.conv2d(Amx0,Wmx,strides = [1,1,1,1],padding = "SAME")
        
        w0 = tf.get_variable(self.name+"w0",shape=[1],initializer=tf.ones_initializer())
        w1 = tf.get_variable(self.name+"w1",shape=[1],initializer=tf.ones_initializer())
        w2 = tf.get_variable(self.name+"w2",shape=[1],initializer=tf.ones_initializer())
        w3 = tf.get_variable(self.name+"w3",shape=[1],initializer=tf.ones_initializer())
        
        self.OUP = A11*w0 + A33*w1 + A55*w2 + Amx*w3
    def __call__(self):
        return self.OUP

class CONV2D:
    def __init__(self,inp,shape,**args):
        try:
            self.padding = args["padding"]
            assert(self.padding in ["SAME","VALID"])
        except:
            self.padding = "VALID"

        try:
            self.strides = args["strides"]
        except:
            self.strides = [1,1,1,1]

        try:
            self.name = args["name"]
        except:
            self.name = "defCONV2D"

        self.shape = shape

        self.W = tf.get_variable(self.name+"W",shape = self.shape,initializer = tf.contrib.layers.xavier_initializer_conv2d())

        self.OUP = tf.nn.conv2d(inp,self.W,strides = self.strides,padding = self.padding)
    def __call__(self):
        return self.OUP

class PLUSB:
    def __init__(self,inp,**args):
        try:
            self.name = args["name"]
        except:
            self.name = "defPLUSB"

        b,w,h,c = inp.shape

        self.b = tf.get_variable(self.name+"b",shape = [w,h,c],initializer = tf.zeros_initializer())

        self.OUP = inp+self.b
    def __call__(self):
        return self.OUP


class GA:
    def __init__(self,inp,**args):
        
        inpw = inp[0]
        if type(inpw) == tf.Tensor:
            b,w,h,c = inp.shape.as_list()
            inpw = tf.reshape(inpw,shape=[w*h,c])
            self.OUP = tf.matmul(tf.transpose(inpw),inpw)
        elif type(inpw) == np.ndarray:
            b,w,h,c = inp.shape
            inpw = inpw.reshape(w*h,c)
            self.OUP = np.matmul(inpw.transpose(),inpw)
        else:
            raise Exception()
    def __call__(self):
        return self.OUP

class MODEL:
    def __init__(self,name="def"):
        self.name = name
        self.X = None
        self.Y = None
        self.OUP = None
        self.init = tf.global_variables_initializer()
        self.cost = None
        self.opt = None
    def open(self,init=True):
        self.sess=tf.Session()
        if init:
            self.sess.run(self.init)
    def close(self,reset=True):
        self.sess.close()
        tf.reset_default_graph()
    def train(self,X,Y,loop=300):
        loss = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()
        for i in range(loop):
            _,cost = sefl.sess.run([self.opt,self.cost],feed_dict={self.X:X,self.Y:Y})
            loss.append(cost)
            if len(loss)>50:
                loss = loss[1:]
            ax.clear()
            ax.plot(loss)
            fig.canvas.draw()
    def train_minib(self,X,Y,dis=False,loop=300,bloop=4,bsize=128):
        loss = []
        L = len(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()
        for i in range(loop):
            idx = np.random.choice(range(L),size = bsize)
            X_ = X[idx,:,:,:]
            Y_ = Y[idx,:,:,:]
            for b in range(bloop):
                _,cost = self.sess.run([self.opt,self.cost],feed_dict={self.X:X_,self.Y:Y_})
                loss.append(cost)
                if len(loss) > 50:
                    loss = loss[1:]
                ax.clear()
                ax.plot(loss)
                fig.canvas.draw()
    def save(self,name = None):
        if name == None:
            name = self.name
        sv = tf.train.Saver()
        sv.save(self.sess,name)
    def load(self,name = None):
        if name == None:
            name = self.name
        sv = tf.train.Saver()
        sv.restore(self.sess,name)


