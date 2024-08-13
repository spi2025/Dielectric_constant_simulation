#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
import time
from raw_data import *
import YUCE
import numpy as np
import pandas as pd
import pylab
from tkinter import filedialog


LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name


    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("介电函数预测_v2.1sx")           #窗口名                       
        self.init_window_name.geometry('620x420+10+20')          # '620x500+10+20' WxH为窗口宽高，+x +y 定义窗口弹出时的默认展示位置(x,y)
        self.init_window_name.resizable(0,0)
        self.init_window_name["bg"] = "PapayaWhip"                 #窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        self.init_window_name.attributes("-alpha",0.9)                          #虚化，值越小虚化程度越高
        self.init_window_name.grid_columnconfigure(0, weight=1)
        self.init_window_name.grid_rowconfigure(0, weight=1)
        self.init_window_name.grid_rowconfigure(1, weight=1)

        #标签1 
        self.frame1 = Frame(self.init_window_name, width=620, height=300, bg='black', bd=5)
        self.frame1.grid(row=0, column=0,sticky = N + S + E + W)
        self.frame1.grid_columnconfigure(0, weight=1)
        self.frame1.grid_columnconfigure(1, weight=1)
        self.frame1.grid_rowconfigure(0, weight=1)

        self.frame11 = Frame(self.frame1, width=62*5, height=300, bg='blue', bd=5)
        self.frame11.grid(row=0, column=0,sticky = N + S + E + W)
        self.train_label = Label(self.frame11, text='训 练 区', width=15, justify=LEFT, relief=RIDGE, background='#6699ff', font=('黑体', 15) )
        self.train_label.pack_configure(anchor=S, side=TOP, fill=NONE, padx=2, pady=2, ipadx=2, ipady=2)
        self.frame111 = Frame(self.frame11, width=62*5, height=2, bg='white', bd=5)
        self.frame111.pack_configure(anchor=S, side=TOP, fill='x', padx=2, pady=2, ipadx=2, ipady=2)
        self.shurushuju_button = Button(self.frame11, text="输入数据文件: (*.xls/*.xlsx)", bg="lightblue", \
                                        width=30,command=self.input_data, justify=LEFT, font=('宋体', 10))
        self.shurushuju_button.pack(fill = NONE,expand = 0, anchor=W, padx=1, pady=1, ipadx=1, ipady=1)
        self.shurushuju_Text = Entry(self.frame11, bd=5, width=10, justify=LEFT)  #  代码录入框
        self.shurushuju_Text.pack(fill = "x",expand = 1, padx=6, pady=1, ipadx=2, ipady=2)
        self.shurushuju_Text.config(state='disabled')
        self.train_button = Button(self.frame11, text="训  练", bg="lightblue", width=10, height=1,command=self.train, justify=LEFT, font=('宋体', 10)) 
        self.train_button.pack(fill = NONE, expand = 0, anchor=W, padx=1, pady=1, ipadx=1, ipady=1) 
        self.trainvalue_Text = Text(self.frame11, width=10, height=16)  #运行结果
        self.trainvalue_Text.pack(fill = BOTH, expand = 1)
        self.trainvalue_Text.config(state='disabled')
        
        self.frame12 = Frame(self.frame1, width=62*5, height=300, bg='red', bd=5)
        self.frame12.grid(row=0, column=1,sticky = N + S + E + W)
        self.yuce_label = Label(self.frame12, text='预 测 区', width=15, justify=LEFT, relief=RIDGE, background='#6699ff', font=('黑体', 15) )
        self.yuce_label.pack_configure(anchor=S, side=TOP, fill=NONE, padx=2, pady=2, ipadx=2, ipady=2)
        self.frame121 = Frame(self.frame12, width=62*5, height=2, bg='white', bd=5)
        self.frame121.pack_configure(anchor=S, side=TOP, fill='x', padx=2, pady=2, ipadx=2, ipady=2)
        self.frame122 = Frame(self.frame12, width=62*5, height=15, bg='blue', bd=5)
        self.frame122.pack_configure(anchor=S, side=TOP, fill='x', padx=2, pady=2, ipadx=2, ipady=2)
        self.cailiao_label = Label(self.frame122, text="材料", bg='lightblue', width=3, font=('宋体', 10))
        self.cailiao_label.pack(side=LEFT, anchor=N, padx=2, pady=1, ipadx=2, ipady=1)
        self.Al_label = Label(self.frame122, text="Al", bg='aqua', width=2, font=('宋体', 12))
        self.Al_label.pack(side=LEFT, anchor=N, padx=2, pady=2, ipadx=2, ipady=1)
        Al_variable = StringVar()
        self.Al_Text = Entry(self.frame122, bd=5, width=4, justify=LEFT, font=('宋体', 10),  \
                             textvariable = Al_variable, validate = "key", validatecommand = (self.frame122.register(self.input01_Al), "%P"))  #  Al
        self.Al_Text.pack(side=LEFT, anchor=N, padx=1, pady=1, ipadx=1, ipady=1)
        self.Ga_label = Label(self.frame122, text="Ga", bg='aqua', width=2, font=('宋体', 12))
        self.Ga_label.pack(side=LEFT, anchor=N, padx=2, pady=2, ipadx=2, ipady=1)
        Ga_variable = StringVar()
        self.Ga_Text = Entry(self.frame122, bd=5, width=4, justify=LEFT, font=('宋体', 10),  \
                             textvariable = Ga_variable, validate = "key", validatecommand = (self.frame122.register(self.input01_Ga), "%P"))  #  Ga
        self.Ga_Text.pack(side=LEFT, anchor=N, padx=1, pady=1, ipadx=1, ipady=1)
        self.In_label = Label(self.frame122, text="In", bg='aqua', width=2, font=('宋体', 12))
        self.In_label.pack(side=LEFT, anchor=N, padx=2, pady=2, ipadx=2, ipady=1)
        self.In_Text = Entry(self.frame122, bd=5, width=4, justify=LEFT, font=('宋体', 10))  #  In
        self.In_Text.pack(side=LEFT, anchor=N, padx=1, pady=1, ipadx=1, ipady=1)
        self.In_Text.config(state='disabled')
        self.N_label = Label(self.frame122, text="N", bg='aqua', width=2, font=('宋体', 12))
        self.N_label.pack(side=LEFT, anchor=N, padx=2, pady=2, ipadx=2, ipady=1)
        self.frame123 = Frame(self.frame12, width=62*5, height=15, bg='blue', bd=5)
        self.frame123.pack_configure(anchor=S, side=TOP, fill='x', padx=2, pady=2, ipadx=2, ipady=2)
        self.energy_label = Label(self.frame123, text="能量范围(eV):", bg='lightblue', width=12, font=('宋体', 10))
        self.energy_label.pack(side=LEFT, anchor=N, padx=2, pady=1, ipadx=2, ipady=1)
        start_variable = StringVar()
        self.start_Text = Entry(self.frame123, bd=5, width=7, justify=LEFT, font=('宋体', 10), \
                             textvariable = start_variable, validate = "key", validatecommand = (self.frame123.register(self.inputdayu0), "%P"))  # start 
        self.start_Text.pack(side=LEFT, anchor=N, padx=1, pady=1, ipadx=1, ipady=1)
        self.jian_label = Label(self.frame123, text="====>", bg='aqua', width=4, font=('宋体', 12))
        self.jian_label.pack(side=LEFT, anchor=N, padx=2, pady=2, ipadx=2, ipady=1)
        end_variable = StringVar()
        self.end_Text = Entry(self.frame123, bd=5, width=7, justify=LEFT, font=('宋体', 10), \
                             textvariable = end_variable, validate = "key", validatecommand = (self.frame123.register(self.inputdayu0), "%P"))  # end
        self.end_Text.pack(side=LEFT, anchor=N, padx=1, pady=1, ipadx=1, ipady=1)
        self.yuce_button = Button(self.frame12, text="预 测", bg="lightblue", width=10, height=1,command=self.yuce, justify=LEFT, font=('宋体', 10)) 
        self.yuce_button.pack(fill = NONE, expand = 0, anchor=W, padx=1, pady=1, ipadx=1, ipady=1)
        self.save_button = Button(self.frame12, text="保存数据", bg="lightblue", width=10, height=1,command=self.save, justify=LEFT, font=('宋体', 10)) 
        self.save_button.pack(fill = NONE, expand = 0, anchor=W, padx=1, pady=1, ipadx=1, ipady=1) 
        
         #标签2
        self.frame2 = Frame(self.init_window_name, width=620, height=200, bg='aqua', bd=5)
        self.frame2.grid(row=2, column=0,sticky = N + S + E + W)
         
        self.log_label= Label(self.frame2, text="日   志", bg='aqua',  fg='maroon')
        self.log_label.pack()
        self.log_data_Text = Text(self.frame2, width=80, height=6)  # 日志框
        self.log_data_Text.pack(fill = BOTH, expand = 1)
        self.log_data_Text.config(state='disabled')

    def input01_Al(self, s):
        d  = re.match(r"[+]?[0]?\.\d*|[0, 1]?",s)
        if d:
            if d.group() == s:
                if s == '':
                    Al = 0
                else:
                    Al = float(s)   
                Ga = self.Ga_Text.get()
                if Ga == '':
                    Ga = 0
                else:
                    Ga = float(Ga) 
                if Ga + Al > 1:
                    return  False
                else:
                    self.In_Text.config(state='normal')
                    v = 1 - Ga - Al
                    self.In_Text.delete(0, 'end')
                    self.In_Text.insert(0, v)
                    self.In_Text.config(state='disabled')      
            return d.group() == s
        return  False

    def input01_Ga(self, s):
        d  = re.match(r"[+]?[0]?\.\d*|[0, 1]?",s)
        if d:
            if d.group() == s:
                if s == '':
                    Ga = 0
                else:
                    Ga = float(s)
                Al = self.Al_Text.get()
                if Al == '':
                    Al = 0
                else:
                    Al = float(Al)
                if Al + Ga > 1:
                    return  False
                else:
                    self.In_Text.config(state='normal')
                    v = 1 - Al - Ga
                    self.In_Text.delete(0, 'end')
                    self.In_Text.insert(0, v)
                    self.In_Text.config(state='disabled')      
            return d.group() == s
        return  False

    def inputdayu0(self, s):
        return re.match(r"[+]?\d*\.?\d*", s).group() == s
        
    def input_data(self):
        self.shurushuju_Text.config(state='normal')
        self.shurushuju_Text.delete(0, 'end')
        text = filedialog.askopenfilename(title = "请选择一个要打开的Excel文件", filetypes = \
                                          [("Microsoft Excel 97-20003 文件", "*.xls"), ("Microsoft Excel文件", "*.xlsx")])
        self.shurushuju_Text.insert(0, text)
        self.shurushuju_Text.config(state='disabled')
        self.write_log_to_Text('INFO: 导入了数据路径')

    def train(self):
        link = self.shurushuju_Text.get()
        if not link:
            msg = '请导入数据路径'
            self.write_trainvalue_Text(msg)
            self.write_log_to_Text('INFO: （出错）未导入数据路径')
        else:
            try:
                df = DF(link)
                self.write_log_to_Text('INFO: 导入数据成功')
            except:
                self.write_log_to_Text('INFO: 导入数据出错')
            try:
                self.write_log_to_Text("INFO:训练开始")
                df_num = len(df.n)
                data = Data(df, df_num)
                result = YUCE.main()
                result.canshu_load('')
                count = 0
                while count < 1000:
                    time_list = 18
                    train_data, price = data.data_set_list(time_list)
                    result.train_c(count, train_data, price, 10)
                    loss = result.losslayer.loss
                    msg = 'loss: %.2f\n'%loss
                    self.write_trainvalue_Text(msg)
                    count += 1
                result.canshu_save('')
                self.write_log_to_Text("INFO:训练完成")
            except:
                self.write_log_to_Text("INFO:训练失败")
            try:
                self.write_log_to_Text("INFO:验证开始")
                validate_data = data.data_set(1000)
                accu = result.validate_c(validate_data)
                msg = '正确率: %.2f\n'%accu
                self.write_trainvalue_Text(msg)
                self.write_log_to_Text("INFO:验证结束")
            except:
                self.write_log_to_Text("INFO:验证失败")
            
    def yuce(self):
        Al = self.Al_Text.get()
        Ga = self.Ga_Text.get()
        In = self.In_Text.get()
        start = self.start_Text.get()
        end = self.end_Text.get()
        if '' in (Al, Ga, In, start, end):
            msg = '请正确输入AlGaInN组分或能量范围\n'
            self.write_trainvalue_Text(msg)
            self.write_log_to_Text('INFO: （出错）未输入数据')
        else:
            try:
                self.write_log_to_Text("INFO:预测开始")
                energy = np.arange(min([float(start), float(end)]), max([float(start), float(end)]), 0.01)
                L = len(energy)
                yuce_data =  np.zeros((L,1,4,1))
                for i in range(L):
                    yuce_data[i,0,:,0] = np.array([float(Al), float(Ga), float(In), energy[i]])
                result = YUCE.main()
                result.canshu_load('')
                x = result.yuce_c(yuce_data)
                x = x.reshape(L, 2)
                self.EE = np.zeros(L)
                self.n = np.zeros(L)
                self.k = np.zeros(L)
                self.EE[:] = energy
                self.n[:] = x[:, 0]
                self.k[:] = x[:, 1]
                pylab.figure(1)
                pylab.plot(self.EE, self.n,'ro--', label="epsilon1")
                pylab.plot(self.EE, self.k,'bo--', label="epsilon2")
                pylab.legend()
                pylab.show()
                self.write_log_to_Text("INFO:预测结束")
            except:
                self.write_log_to_Text("INFO:预测失败")
                
    def save(self):
        try:
            data_array = []
            data_array.append(self.EE)
            data_array.append(self.n)
            data_array.append(self.k)
            epsilon = self.n + 1j*self.k
            nk = np.sqrt(epsilon)
            data_array.append(np.real(nk))
            data_array.append(np.imag(nk))
            np_data = np.array(data_array)
            np_data = np_data.T
            save = pd.DataFrame(np_data, columns = ['Energy (eV)', 'epsilon1', 'epsilon2', 'n', 'k'])
            link = filedialog.asksaveasfilename(title = "请创建或者选择一个保存数据的Excel文件", filetypes = [("Microsoft Excel 逗号分隔文件", "*.csv")], \
                                                defaultextension = ".xls")
            save.to_csv(link, index=False, header=True)
            self.write_log_to_Text("INFO:保存完成")
        except:
            self.write_log_to_Text("INFO:保存失败")

    def write_trainvalue_Text(self, msg):
        self.trainvalue_Text.config(state='normal')
        self.trainvalue_Text.insert(END, msg)
        self.trainvalue_Text.update()
        self.trainvalue_Text.config(state='disabled')
    #获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return current_time

     #日志动态打印
    def write_log_to_Text(self,logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) +" " + str(logmsg) + "\n"      #换行
        self.log_data_Text.config(state='normal')
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(END, logmsg_in)
            self.log_data_Text.update()
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0,2.0)
            self.log_data_Text.insert(END, logmsg_in)
            self.log_data_Text.update()
        self.log_data_Text.config(state='disabled')

class DF:
    def __init__(self,link):
        self.Al = []
        self.Ga = []
        self.In = []
        self.EE = []
        self.n = []
        self.k = []

        try:
            df = pd.read_csv( link, header=None,encoding='gb18030')
        except:
            df = pd.read_excel(link)
        
        for i in range(0,len(df.values[:,1])):
            self.Al.append(float(df.values[i,0]))
            self.Ga.append(float(df.values[i,1]))
            self.In.append(float(df.values[i,2]))
            self.EE.append(float(df.values[i,3]))
            self.n.append(float(df.values[i,4]))
            self.k.append(float(df.values[i,5]))

        self.Al = np.array(self.Al)
        self.Ga = np.array(self.Ga)
        self.In = np.array(self.In)
        self.EE = np.array(self.EE)
        self.n = np.array(self.n)
        self.k = np.array(self.k)

def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()
    

    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示


gui_start()

