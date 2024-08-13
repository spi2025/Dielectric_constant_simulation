import numpy as np
class Data:
    def __init__(self, df, df_num):
        '''
        # df      : 输入数据
        # df_num  : 数据时间长度
        '''
        self.df = df
        self.num_1 = df_num
        self.xulie = np.arange(self.num_1)
        
    def data_samp_24(self):
        
        data_set0 = []   # 数据
        data_set0s = []   # 数据
        
        for i in range(self.num_1):
            data_set1 = []
            data_set2 = []

            data_set1.append(self.df.Al[i])
            data_set1.append(self.df.Ga[i])
            data_set1.append(self.df.In[i])
            data_set1.append(self.df.EE[i])
            data_set2.append(self.df.n[i])
            data_set2.append(self.df.k[i])
            
            data_set0.append(np.array(data_set1))
            data_set0s.append(np.array(data_set2))
            
        data_set0 = np.array(data_set0)
        data_set0s = np.array(data_set0s)
        
        return data_set0, data_set0s
            
    def data_set(self, num):
        np.random.shuffle(self.xulie)
        time_list = self.xulie[0:num]
        D_num = 0
        data_set0, label_set0 = self.data_samp_24()
        data_4D = np.zeros((len(time_list),1,4,1))
        label = np.zeros((len(time_list),2))
        
        for i in time_list:
            label[D_num,:] = label_set0[i,:]
            data_4D[D_num,0,:,0] = data_set0[i,:]
            D_num += 1
            
        return data_4D, label

    def data_set_list(self, num):
        np.random.shuffle(self.xulie)
        data_set0, label_set0 = self.data_samp_24()
        shumu = int(self.num_1/num)
        data_4D_list = []
        label_list = []
        for ii in range(num):
            time_list = self.xulie[ii*shumu:(ii+1)*shumu]
            D_num = 0
            data_4D = np.zeros((len(time_list),1,4,1))
            label = np.zeros((len(time_list),2))
            for i in time_list:
                label[D_num,:] = label_set0[i,:]
                data_4D[D_num,0,:,0] = data_set0[i,:]
                D_num += 1
            data_4D_list.append(data_4D)
            label_list.append(label)
        return data_4D_list, label_list
