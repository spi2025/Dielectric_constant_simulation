import numpy as np
import time

class Conv_Connect:
    def __init__(self, pad="same", w_shape=(3,3,3,16), stride=1, dilation=1):
        """
        函数说明：
        :param pad: （可取 int, tuple, "same", "valid"） "same" 输出输入尺寸相同； "valid"不扩张；tuple （上，下，左，右）
        :param w_shape: (卷积核大小fr fc 输入通道in_ch 输出通道out_ch)
        :param stride:  步长
        :param dilation: 空洞因子
        """
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        
        #self.weights = np.random.random(size = w_shape)
        #self.bias = np.random.random(size =(w_shape[3], 1))
        self.weights = 0.01*np.random.randn(w_shape[0],w_shape[1],w_shape[2],w_shape[3])
        self.bias = 0.01*np.random.randn(w_shape[3], 1)

        self.weights_x  = 0
        self.bias_x = 0
        self.E_x_x = 0
        self.Var_x_x  = 0
        
        self.lrw = 0.01
        self.lrb = 0.01
        self.db_x = 0
        self.dw_x = 0

        """ 定义一个pading计算函数，可以使输入和输出的特征图尺度大小一致"""
    def caculate_paddims_same_2Dconv(self,X_shape, out_dim, kernel_shape, stride, dilation=1):
        """
        param :
        :param X_shape: 输入特征图的尺度
        :param out_dim: 输出特征图尺度
        :param kernel_shape: 卷积核大小
        :param stride:
        :param dilation: 空洞因子
        :return: 上下左右的pading值。
        """
        d = dilation
        fr, fc = kernel_shape
        out_r, out_c = out_dim
        nex, in_r, in_c, in_chanle = X_shape # 分表表示输入输出以及通道值。
        # 先计算 空洞卷积的空洞卷积值。
        fr_x, fc_x = fr + (fr-1)*(d-1), fc + (fc-1)*(d-1)
        # compute pad value.
        # formuler : [w + 2p - (k+(k-1)*(d-1))]/s +1 = outw
        # p = [(outw -1) * s + fr_x - w ]/2
        pr,pc =int( ((out_r -1) * stride + fr_x - in_r )/2), int( ((out_c -1) * stride + fc_x - in_c )/2)

        # 上边得到的pr,pc 可能会小一个，因为如果需要扩张的pading总数为奇数，两边分一下就成为了0.5，取整后，就舍弃了
        # 一个pad单位，所以需要进行验证和修正。
        # use pr,pc compute the outsize, and test the outsize is eq with the needed outsize.
        n_out_r, n_out_c = int((in_r+ 2 * pr - fr_x + stride)/stride), int((in_c+ 2 * pc - fc_x + stride)/stride)

        # panduan
        pr1,pr2 = pr,pr # 上下的扩张大小。
        if n_out_r == out_r -1 :# 如果相等就直接返回了，如果差1就在下边加上一个，否则报错。
            pr1,pr2 = pr, pr+1
        elif n_out_r != out_r:
            raise AssertionError  # 声明错误。

        pc1, pc2 = pc, pc  # 左右的扩张大小。
        if n_out_c == out_c - 1:  # 如果相等就直接返回了，如果差1就在下边加上一个，否则报错。
            pc1, pc2 = pc, pc + 1
        elif n_out_c != out_c:
            raise AssertionError  # 声明错误。

        return  (pr1,pr2,pc1,pc2) # 上下，左右的扩张值。


    """  对输入X 进行pading, X .shape : batchsize, h,w, num_channel"""
    def Pad2D(self, X, pad, kernel_shape=None, stride=None, dilation=1):
        """
        函数说明：对输入X 进行pading
        :param X:  shape : bz, h,w,n_c
        :param pad: int tuple, "same", "valid"
        :param kernel_shape:
        :param stride:
        :param dilation:
        :return:
        """
        p = pad
        if isinstance(p, int):
            p = (pad,pad,pad,pad)

        if isinstance(p, tuple):
            X_pad = np.pad(X,
                           pad_width=((0,0), (p[0], p[1]), (p[2],p[3]),(0,0)),
                           mode="constant",
                           constant_values=0,
                           )

        # 如果pad 是字符 same,或者valid
        if pad == "same" and kernel_shape and stride is not None:
            p = self.caculate_paddims_same_2Dconv(X.shape, X.shape[1:3], kernel_shape, stride, dilation)
            X_pad, p = self.Pad2D(X,p) # 这里使用递归调用方法。

        if pad == "valid":
            p = (0,0,0,0)
            X_pad , p = self.Pad2D(X,p)

        return X_pad, p

    
    """  用矩阵的乘法来进行卷积的运算"""
    def _im2col_indices(self, x_shape, fr, fc, p, s, d=1):
        """ 计算各个索引"""
        pr1, pr2, pc1, pc2 = p
        n_ex, in_rows, in_cols, n_in = x_shape
        _fr, _fc = fr + (fr - 1) * (d - 1), fc + (fc - 1) * (d - 1)

        out_rows = int((in_rows + pr1 + pr2 - _fr + s) / s)
        out_cols = int((in_cols + pc1 + pc2 - _fc + s) / s)
        #print(out_rows,out_cols)
        # 28 28
        i0 = np.repeat(np.arange(fr), fc)  # 000111222   * n_in.
        # 000111222
        i0 = np.tile(i0, n_in) * d

        i1 = s * np.repeat(np.arange(out_rows), out_cols)  # 00000..0 11111..1 2222..2.
        # 这里i1 的个数其实就是输出的图像的尺度的长宽大小。
        # 对于每一个位置，都需要相应的卷积得到结果。
        j0 = np.tile(np.arange(fc), fr * n_in)* d  # 相当与相对索引。
        j1 = s * np.tile(np.arange(out_cols), out_rows) # 相当于绝对索引。 i1 j1 确定位置， i0，j0 确定卷积。得到切块。
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        # 第二个的索引。
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        # 第三个索引。            
        k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
        return k, i, j

    def im2col(self,X, W_shape, pad, stride, dilation=1):
        fr, fc, n_in, n_out = W_shape
        s, p, d = stride, pad, dilation
        n_samp, in_rows, in_cols, n_in = X.shape
        X_pad, p = self.Pad2D(X, p, W_shape[:2], stride=s, dilation=d)
        pr1, pr2, pc1, pc2 = p
        # 将输入的通道维数移至第二位
        X_pad = X_pad.transpose(0, 3, 1, 2)
        k, i, j = self._im2col_indices((n_samp, in_rows, in_cols, n_in), fr, fc, p, s, d)
        # X_col.shape = (n_samples, kernel_rows*kernel_cols*n_in, out_rows*out_cols)
        X_col = X_pad[:, k, i, j]  # i,j,k 联合位置的元素值。形状与i，j，k 形状有关。
        X_col = X_col.transpose(1, 2,0).reshape(fr * fc * n_in, -1)
        return X_col, p

    def forward(self, X):  # 输入数据 (n_samp, in_rows, in_cols, n_in)
        self.x = X
        W = self.weights
        s, d = self.stride, self.dilation
        _, p = self.Pad2D(X, self.pad, W.shape[:2], s, d)
        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = W.shape
        n_samp, in_rows, in_cols, in_ch = X.shape
        # 考虑扩张率
        _fr, _fc = fr + (fr - 1) * (d - 1), fc + (fc - 1) * (d - 1)
        # 输出维数，根据上面公式可得
        out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)
        # 将 X 和 W 转化为 2D 矩阵并乘积
        X_col, _ = self.im2col(X, W.shape, p, s, d) # 转换的输入数据X_col  ( in_ch*fr * fc, out_rows*out_cols*n_samp)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)
        
        b_col = np.repeat(self.bias,out_rows*out_cols*n_samp).reshape(out_ch,out_rows*out_cols*n_samp)
        # 输出数据 (n_samp, out_rows, out_cols, out_ch)
        Z = (W_col @ X_col+b_col).reshape(out_ch, out_rows, out_cols, n_samp).transpose(3, 1, 2, 0)
        return Z

    def backward(self, d_p):
        W = self.weights
        fr, fc, in_ch, out_ch = W.shape
        s, d = self.stride, self.dilation
        _, p = self.Pad2D(self.x, self.pad, W.shape[:2], s, d)
        pr1, pr2, pc1, pc2 = p
        n_samp, in_rows, in_cols, in_ch = self.x.shape
        # 考虑扩张率
        _fr, _fc = fr + (fr - 1) * (d - 1), fc + (fc - 1) * (d - 1)
        # 输出维数，根据上面公式可得
        out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

        X_col, _ = self.im2col(self.x, W.shape, p, s, d)
        dd = d_p.transpose(3,1,2,0).reshape(out_ch,-1)
        xx = ((X_col).T)/(out_rows*out_cols*n_samp)
        dw = (dd @ xx).reshape(out_ch,in_ch,fr,fc).transpose(2,3,1,0)
        
        db = np.sum(dd,axis=1).reshape(out_ch,1)/(out_rows*out_cols*n_samp)
        
        
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)
        dX_col = (W_col.T @ dd).reshape(fr * fc * in_ch, out_rows*out_cols, n_samp).transpose(2, 0, 1)
        dx = np.zeros((n_samp, in_rows+pr1+pr2, in_cols+pc1+pc2, in_ch))     
        k, i, j = self._im2col_indices((n_samp, in_rows, in_cols, in_ch), fr, fc, p, s, d)

        dx = dx.transpose(0, 3, 1, 2)
        dx[:, k, i, j] += dX_col
        dx = dx.transpose(0, 2, 3, 1)
        
        dx = np.delete(dx, range(0,pr1),1)
        dx = np.delete(dx, range(-1,-pr2-1,-1),1)
        dx = np.delete(dx, range(0,pc1),2)
        dx = np.delete(dx, range(-1,-pc2-1,-1),2)

        #self.lrb = 2/(1+np.exp(-np.abs(db)*10**1))-1
        #self.lrw = 2/(1+np.exp(-np.abs(dw)*10**1))-1

        a_ = 0.3
        db_ = (1-a_)*db + a_*self.db_x
        dw_ = (1-a_)*dw + a_*self.dw_x
        
        self.bias -= self.lrb*db_
        self.weights -= self.lrw*dw_
        
        self.db_x = db_
        self.dw_x = dw_
        
        return dx

class Pooling_MAX:
    def __init__(self, FF=(2,2), SS=2):
        '''
        函数说明：
        :param FF: 空间大小
        :param SS: 步长
        '''
        self.FF = FF
        self.SS = SS

    def _im2col_indices(self, x_shape, FF, SS):
        """ 计算各个索引"""
        FF1, FF2 = FF
        n_ex, in_rows, in_cols, n_in = x_shape
        out_rows = int((in_rows - FF1 + SS) / SS)
        out_cols = int((in_cols - FF2 + SS) / SS)
        
        i0 = np.repeat(np.arange(FF1), FF2)  # 000111222 
        
        i1 = SS * np.repeat(np.arange(out_rows), out_cols)  # 00000..0 11111..1 2222..2.
        # 这里i1 的个数其实就是输出的图像的尺度的长宽大小。
        # 对于每一个位置，都需要相应的卷积得到结果。
        j0 = np.tile(np.arange(FF2), FF1)  # 相当与相对索引。
        j1 = SS * np.tile(np.arange(out_cols), out_rows) # 相当于绝对索引。 i1 j1 确定位置， i0，j0 确定卷积。得到切块。
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        # 第二个的索引。
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        return i, j

    def im2col(self, X, FF, SS):
        F1, F2 = FF
        n_samp, in_rows, in_cols, n_in = X.shape
        # 将输入的通道维数移至第二位
        X_pad = X.transpose(0, 3, 1, 2)
        i, j = self._im2col_indices((n_samp, in_rows, in_cols, n_in), FF, SS)
        # X_col.shape = (n_samples, n_in, F1_rows*F2_cols, out_rows*out_cols)
        X_col = X_pad[:, :, i, j]  
        X_col = X_col.transpose(2, 3, 1, 0).reshape(F1*F2, -1)
        return X_col

    def forward(self, X):  # 输入数据 (n_samp, in_rows, in_cols, n_in)
        self.x = X
        F1, F2 = self.FF
        S = self.SS
        n_samp, in_rows, in_cols, in_ch = X.shape
        # 输出维数，根据上面公式可得
        out_rows = int((in_rows - F1 + S) / S)
        out_cols = int((in_cols - F2 + S) / S)
        
        # 将 X 转化为 2D 矩阵
        X_col= self.im2col(X, (F1, F2), S) # 转换的输入数据X_col  ( F1_rows*F2_cols, out_rows*out_cols*in_ch*n_samp)
        
        # 输出数据 (n_samp, out_rows, out_cols, out_ch=in_ch)
        Z = (np.max(X_col,axis=0)).reshape(out_rows, out_cols, in_ch, n_samp).transpose(3, 0, 1, 2)
        self.Z_index = np.where(X_col==np.max(X_col,axis=0))
        return Z

    def backward(self, d_p):
        F1, F2 = self.FF
        S = self.SS
        n_samp, in_rows, in_cols, in_ch = self.x.shape
        # 输出维数，根据上面公式可得
        out_rows = int((in_rows - F1 + S) / S)
        out_cols = int((in_cols - F2 + S) / S)
        
        dd = d_p.transpose(1,2,3,0).reshape(out_rows*out_cols*in_ch*n_samp)
        ddx = np.zeros((n_samp, in_rows, in_cols, in_ch))
        ddx = self.im2col(ddx, (F1, F2), S)
        ddx[self.Z_index[0],self.Z_index[1]] += dd[self.Z_index[1]]
        ddx = ddx.reshape(F1*F2, out_rows*out_cols, in_ch, n_samp).transpose(3, 2, 0, 1)
        
        i, j = self._im2col_indices((n_samp, in_rows, in_cols, in_ch), (F1,F2), S)
        dx = np.zeros((n_samp, in_rows, in_cols, in_ch))
        dx = dx.transpose(0, 3, 1, 2)
        dx[:, :, i, j] += ddx
        dx = dx.transpose(0, 2, 3, 1)

        return dx

class Batch_Norm:
    def __init__(self, l_x):
        self.weights  = 0.01*np.random.randn(l_x[0],l_x[1],l_x[2])
        self.bias = 0.01*np.random.randn(l_x[0],l_x[1],l_x[2])
        self.E_x = 0.5*np.ones((l_x[0],l_x[1],l_x[2]))
        self.Var_x = np.zeros((l_x[0],l_x[1],l_x[2]))
        
        self.weights_x  = 0
        self.bias_x = 0
        self.E_x_x = 0
        self.Var_x_x  = 0
        
        self.lrb = 0.01
        self.lrw =0.01
        self.db_x = 0
        self.dw_x = 0

    def E_Var(self, x):
        mu = np.sum(x, axis=0)/x.shape[0]
        sigma2 = np.sum(np.square(x[:]-mu), axis=0)/(x.shape[0]-1)
        a = 0.9
        self.E_x =  a*self.E_x + (1-a)*mu
        self.Var_x = a*self.Var_x + (1-a)*sigma2
        
    def forward(self, x):
        w = self.weights
        self.x = x
        y1 = (x*w)/np.sqrt(self.Var_x+np.exp(-30))
        y2 = self.bias - (self.E_x*w)/np.sqrt(self.Var_x+np.exp(-30))
        y = y1 + y2
        return y

    def backward(self, d):
        w = self.weights
        ddw = (d*self.x)/np.sqrt(self.Var_x+np.exp(-30))
        dw = np.sum(ddw,axis=0)/self.x.shape[0]
        db = np.sum(d,axis=0)/self.x.shape[0]
        dx = (d*w)/np.sqrt(self.Var_x+np.exp(-30))
        
        #self.lrb = 2/(1+np.exp(-np.abs(db)*10**1))-1
        #self.lrw = 2/(1+np.exp(-np.abs(dw)*10**1))-1

        a_ = 0.3
        db_ = (1-a_)*db + a_*self.db_x
        dw_ = (1-a_)*dw + a_*self.dw_x
        
        self.weights -= self.lrw*dw_
        self.bias -= self.lrb*db_
        
        self.db_x = db_
        self.dw_x = dw_
        
        return dx

class Batch_Norm_0:
    def __init__(self):  
        self.weights  = np.zeros(1)
        self.bias = np.zeros(1)
        
        self.weights_x  = 0
        self.bias_x = 0
        self.E_x_x = 0
        self.Var_x_x  = 0

    def E_Var(self, x):
        self.E_x = np.sum(x, axis=0)/x.shape[0]
        self.Var_x = np.sum(np.square(x[:]-self.E_x), axis=0)/(x.shape[0]-1)
        
    def forward(self, x):
        self.x = x
        y1 = x/np.sqrt(self.Var_x+np.exp(-30))
        y2 = - self.E_x/np.sqrt(self.Var_x+np.exp(-30))
        y = y1 + y2
        return y

    def backward(self, d):
        pass
    
class Sig_norm:
    def __init__(self):
        pass

    def sigmoid(self, x):
        a = (x+abs(x))/2
        a_sig = 1/(1 + np.exp(-a))
        b = (x-abs(x))/2
        b_sig = np.exp(b)/(1 + np.exp(b))
        return a_sig+b_sig-0.5

    def forward(self, x):
        self.x = self.sigmoid(x)
        n_samp, in_rows, in_cols, in_ch = self.x.shape
        x_2d = self.x.reshape(n_samp*in_rows*in_cols,in_ch)
        x_2d = x_2d**2
        x_2d_sum = np.repeat(np.sum(x_2d,axis=1),in_ch).reshape(n_samp*in_rows*in_cols,in_ch)
        x_ = (x_2d/(x_2d_sum+np.exp(-30))).reshape(n_samp, in_rows, in_cols, in_ch)
        return x_

    def backward(self, d):
        n_samp, in_rows, in_cols, in_ch = self.x.shape
        temp_sum = np.sum((self.x**2),axis=3)
        temp_sum = (np.repeat(temp_sum,in_ch)).reshape(n_samp, in_rows, in_cols, in_ch)
        temp = (2*self.x*temp_sum-2*self.x**3)/(temp_sum**2+np.exp(-30))
        temp = temp*self.x*(1-self.x)
        
        return d*temp

class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        a = (x+abs(x))/2
        a_sig = 1/(1 + np.exp(-a))
        b = (x-abs(x))/2
        b_sig = np.exp(b)/(1 + np.exp(b))
        return a_sig+b_sig-0.5

    def forward(self, x):
        self.x = self.sigmoid(x)
        
        return self.x

    def backward(self, d):
        sig = self.x
        
        return d*sig*(1-sig)

class ReLU:
    def __init__(self):
        pass

    def relu(self, x):
        
        return (x + abs(x))/2

    def forward(self, x):
        self.x = x

        return self.relu(x)
    
    def backward(self, d):
        i = np.int64(self.x>0)
        
        return i*d

class PReLU:
    def __init__(self):
        self.alpha = 0.001

    def prelu(self, x):
        x1 = (x+abs(x))/2
        x2 = (x-abs(x))/2
        
        return x1+self.alpha*x2

    def forward(self, x):
        self.x = x
        
        return self.prelu(x)
    
    def backward(self, d):
        x1 = np.int64(self.x>0)
        x2 = np.int64(self.x<0)
        
        return x1*d+x2*d*self.alpha

class PReLU0:
    def __init__(self):
        self.alpha = 0.0

    def prelu(self, x):
        x1 = (x+abs(x))/2
        x2 = (x-abs(x))/2
        
        return x1+self.alpha*x2

    def forward(self, x):
        self.x = x
        
        return self.prelu(x)
    
    def backward(self, d):
        x1 = np.int64(self.x>0)
        x2 = np.int64(self.x<0)
        
        return x1*d+x2*d*self.alpha

class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        L = 0.1
        accuracy = np.sum([ll[0]*(1-L)<xx[0][0][0]<ll[0]*(1+L) and ll[1]*(1-L)<xx[0][0][1]<ll[1]*(1+L) for xx, ll in zip(x, label)])
        accuracy = 1.0*accuracy/x.shape[0]
        return accuracy

class Smooth_L1_Loss:
    def __init__(self, d=0.5):
        self.d = d

    def forward(self, x, label):
        self.x = x
        self.label = label
        n_samp, in_rows, in_cols, n_in = x.shape
        xx = x.reshape(n_samp*in_rows*in_cols,n_in)
        Ldelta = 0.5*(xx-label)**2
        for i in range(n_samp*in_rows*in_cols):
            if abs(xx[i,0]-label[i,0]) > self.d:
                Ldelta[i,0] = self.d*abs(xx[i,0]-label[i,0]) - 0.5*self.d**2
            if abs(xx[i,1]-label[i,1]) > self.d:
                Ldelta[i,1] = self.d*abs(xx[i,1]-label[i,1]) - 0.5*self.d**2

        self.loss = np.sum(Ldelta)/n_samp
        return self.loss      

    def backward(self):
        n_samp, in_rows, in_cols, n_in = self.x.shape
        xx = self.x.reshape(n_samp*in_rows*in_cols,n_in)
        self.dx = xx - self.label
        for i in range(n_samp*in_rows*in_cols):
            if abs(xx[i,0]-self.label[i,0]) > self.d:
                if (xx[i,0]-self.label[i,0]) > 0:
                    self.dx[i,0] = self.d
                else:
                    self.dx[i,0] = -self.d
            if abs(xx[i,1]-self.label[i,1]) > self.d:
                if (xx[i,1]-self.label[i,1]) > 0:
                    self.dx[i,1] = self.d
                else:
                    self.dx[i,1] = -self.d

        self.dx = self.dx.reshape(n_samp, in_rows, in_cols, n_in)
        return 0.1*self.dx


