from CNN_mode import *
import math
import matplotlib.pyplot as plt

class main:
    def __init__(self, in_data=(1,4,1)):
        
        #np.random.seed(0)
        
        self.inner_layers = []
        
        self.inner_layers.append(Conv_Connect(pad = 0, w_shape=(in_data[0], in_data[1], in_data[2], 200)))
        self.inner_layers.append(Batch_Norm((1,1,200)))
        self.inner_layers.append(PReLU())

        self.inner_layers.append(Conv_Connect(pad = 0, w_shape=(1, 1, 200, 800)))
        self.inner_layers.append(Batch_Norm((1,1,800)))
        self.inner_layers.append(PReLU())
        
        self.inner_layers.append(Conv_Connect(pad = 0, w_shape=(1, 1, 800, 400)))
        self.inner_layers.append(Batch_Norm((1,1,400)))
        self.inner_layers.append(PReLU())

        self.inner_layers.append(Conv_Connect(pad = 0, w_shape=(1, 1, 400, 2)))
        #self.inner_layers.append(Batch_Norm((1,1,2)))
        self.inner_layers.append(PReLU0())
        
        self.losslayer = Smooth_L1_Loss()
        self.accuracy = Accuracy()
        
    def fuzhiAB(self, A, B):
        index1 = A.shape
        index2 = B.shape
        L = len(index1)
        if L == 1:
            n1 = np.min([index1[0], index2[0]])
            A[0:n1] = B[0:n1]
        elif L == 2:
            n1 = np.min([index1[0], index2[0]])
            n2 = np.min([index1[1], index2[1]])
            A[0:n1, 0:n2] = B[0:n1, 0:n2]
        elif L == 3:
            n1 = np.min([index1[0], index2[0]])
            n2 = np.min([index1[1], index2[1]])
            n3 = np.min([index1[2], index2[2]])
            A[0:n1, 0:n2, 0:n3] = B[0:n1, 0:n2, 0:n3]
        elif L == 4:
            n1 = np.min([index1[0], index2[0]])
            n2 = np.min([index1[1], index2[1]])
            n3 = np.min([index1[2], index2[2]])
            n4 = np.min([index1[3], index2[3]])
            A[0:n1, 0:n2, 0:n3, 0:n4] = B[0:n1, 0:n2, 0:n3, 0:n4]
        else:
            print('error')
        return A

    def fuzhiABs(self, A, B):
        index1 = A.shape
        index2 = B.shape
        L = len(index1)
        if L == 1:
            n1 = np.max([index1[0], index2[0]])
            temp = np.zeros((n1))
            temp[0:index1[0]] = A[:]
            temp[0:index2[0]] = B[:]
            return temp
        elif L == 2:
            n1 = np.max([index1[0], index2[0]])
            n2 = np.max([index1[1], index2[1]])
            temp = np.zeros((n1, n2))
            temp[0:index1[0], 0:index1[1]] = A[:, :]
            temp[0:index2[0], 0:index2[1]] = B[:, :]
            return temp
        elif L == 3:
            n1 = np.max([index1[0], index2[0]])
            n2 = np.max([index1[1], index2[1]])
            n3 = np.max([index1[2], index2[2]])
            temp = np.zeros((n1, n2, n3))
            temp[0:index1[0], 0:index1[1], 0:index1[2]] = A[:, :, :]
            temp[0:index2[0], 0:index2[1], 0:index2[2]] = B[:, :, :]
            return temp
        elif L == 4:
            n1 = np.max([index1[0], index2[0]])
            n2 = np.max([index1[1], index2[1]])
            n3 = np.max([index1[2], index2[2]])
            n4 = np.max([index1[3], index2[3]])
            temp = np.zeros((n1, n2, n3, n4))
            temp[0:index1[0], 0:index1[1], 0:index1[2], 0:index1[3]] = A[:, :, :, :]
            temp[0:index2[0], 0:index2[1], 0:index2[2], 0:index2[3]] = B[:, :, :, :]
            return temp
        else:
            print('error')
            return B
            
    def canshu_load(self,ID):
        for i in range(0, len(self.inner_layers)):
            try:
                weights = np.load('data_par/weight_'+ID+str(i)+'.npy')
                self.inner_layers[i].weights = self.fuzhiAB(self.inner_layers[i].weights, weights)
                #print(weights.shape)
                bias = np.load('data_par/bias_'+ID+str(i)+'.npy')
                self.inner_layers[i].bias = self.fuzhiAB(self.inner_layers[i].bias, bias)
                #print(bias.shape)
                E_x = np.load('data_par/E_x_'+ID+str(i)+'.npy')
                self.inner_layers[i].E_x = self.fuzhiAB(self.inner_layers[i].E_x, E_x)
                #print(E_x.shape)
                Var_x = np.load('data_par/Var_x_'+ID+str(i)+'.npy')
                self.inner_layers[i].Var_x = self.fuzhiAB(self.inner_layers[i].Var_x, Var_x)
                #print(Var_x.shape)
            except:
                pass
                #print('no file')
        
    def canshu_save(self,ID):
        for i in range(0, len(self.inner_layers)):
            try:
                
                weights = np.load('data_par/weight_'+ID+str(i)+'.npy')
                weights = self.fuzhiABs(weights, self.inner_layers[i].weights)
                np.save('data_par/weight_'+ID+str(i)+'.npy', weights)
                bias = np.load('data_par/bias_'+ID+str(i)+'.npy')
                bias = self.fuzhiABs(bias, self.inner_layers[i].bias)
                np.save('data_par/bias_'+ID+str(i)+'.npy', bias)
                E_x = np.load('data_par/E_x_'+ID+str(i)+'.npy')
                E_x = self.fuzhiABs(E_x, self.inner_layers[i].E_x)
                np.save('data_par/E_x_'+ID+str(i)+'.npy', E_x)
                Var_x = np.load('data_par/Var_x_'+ID+str(i)+'.npy')
                Var_x = self.fuzhiABs(Var_x, self.inner_layers[i].Var_x)
                np.save('data_par/Var_x_'+ID+str(i)+'.npy', Var_x)
            except:
                try:
                    np.save('data_par/weight_'+ID+str(i)+'.npy', self.inner_layers[i].weights)
                    np.save('data_par/bias_'+ID+str(i)+'.npy', self.inner_layers[i].bias)
                    np.save('data_par/E_x_'+ID+str(i)+'.npy', self.inner_layers[i].E_x)
                    np.save('data_par/Var_x_'+ID+str(i)+'.npy', self.inner_layers[i].Var_x)
                except:
                    pass
                
            

    def dropout(self, x, rate=0.001):
        
        return self.get_0_2_array(x, rate)
            
    def get_0_1_array(self, array, rate):
        '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.1'''
        zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
        new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
        new_array[:zeros_num] = 0 #将一部分换为0
        np.random.shuffle(new_array)#将0和1的顺序打乱
        re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
        array *= re_array
        array /= (1-rate)
        return array

    def get_0_2_array(self, x, rate):
        retain_prob = 1. - rate
        n_samp, in_rows, in_cols, in_ch = x.shape
        sample = np.random.binomial(n = 1, p = retain_prob, size = x.shape)
        x *= sample
        x /= retain_prob
        return sample/retain_prob
        #return x

    def train_c(self, epochs, train_data, label, ITERS = 100):
        length = len(train_data)
        for i in range(length):
            lossum = 1*float('inf')
            iters = 0
            temp_0_1 = []
            while iters < ITERS:
                x = train_data[i]
                num =0
                for layer in self.inner_layers:
                    try:
                        layer.E_Var(x)
                    except:
                        pass
                    
                    
                    if not iters:
                        if num in [5,8]:
                            temp_0_1.append(self.dropout(x, rate=0.0001))
                        else:
                            temp_0_1.append(1)
                    x = layer.forward(x*temp_0_1[num])

                    '''
                    if num in [2, 5, 8]:
                        n_samp, in_rows, in_cols, in_ch = x.shape
                        x_tu = x.reshape(n_samp, in_ch)
                        x_1 = np.arange(n_samp)
                        y_1 = np.arange(in_ch)
                        print(x_1.shape)
                        print(x_tu.shape)
                        
                        plt.figure()
                        #x_1, y_1 = np.meshgrid(y_1, x_1)
                        #plt.contourf(x_1, y_1, x_tu, 20, cmap='RdGy')
                        #plt.colorbar()
                        for i_ in range(n_samp):
                            plt.scatter(y_1,x_tu[i_, :],marker='.')
                        plt.show()

                    '''
                        
                    '''
                    
                    if num in [5,8]:
                        x = self.dropout(x, rate=0.05)
                    x = layer.forward(x)
                    '''
                    num += 1
                    #print(x.shape)

                loss = self.losslayer.forward(x, label[i])

                if loss > lossum:
                    break

                d = self.losslayer.backward()

                for layer in self.inner_layers[::-1]:
                    d = layer.backward(d)
                    #print(d)
            
                lossum = loss
                iters += 1
                    
            print('train:: epochs:%d::%d'%(epochs, i), 'iters:', iters, 'loss: ', loss)

    def validate_c(self, validate_data):
        x, label = validate_data
        
        for layer in self.inner_layers:
            x = layer.forward(x)
        accu = self.accuracy.forward(x,label)
        return accu
        #print('validate:: accuracy:%f'%accu)
        #print(x)

    def yuce_c(self, yuce_data):
        x = yuce_data
        for layer in self.inner_layers:
            x = layer.forward(x)

        #print(x)
        return x
