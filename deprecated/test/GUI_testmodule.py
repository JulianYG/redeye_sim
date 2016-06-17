import sys
import Tkinter as tk
import subprocess
import testmodule
import setup

paths = setup.config()
croot = paths.getLoc('caffeRoot')
iroot = paths.getLoc('valDataRoot')

class MyOptionMenu(tk.OptionMenu):
    def __init__(self, master, status, *options):
        self.var = tk.StringVar(master)
        self.var.set(status)
        tk.OptionMenu.__init__(self, master, self.var, *options)
        self.config(
            font=('calibri', (10)), bg='white', width=20, fg='dark red')
        self['menu'].config(font=('calibri', (10)), bg='white', fg='dark blue')

       #tk.Entry.__init__(parent, textvariable=username)

    def callback(self):
        val = '{}'.format(self.var.get())
        print(val)
        # subprocess.call([val])


def submit_data2():
    val = mymenu2.get()
    print val

def submit_data3():
    val = mymenu3.get()
    print val

def submit_data4():
    val = mymenu4.get()
    print val

def submit_data5():
    val = mymenu5.get()
    print val

def submit_data7():
    val = mymenu7.get()
    print val

def submit_data8():
    val = mymenu8.get()
    print val

def submit_data9():
    val = mymenu9.get()
    print val

def submit_data10():
    val = mymenu10.get()
    print val

class AllOptions():
     def __init__(self, mymenu1, mymenu2, mymenu3, mymenu4, mymenu5, mymenu6, mymenu7, mymenu8, mymenu9,mymenu10):
         self.mymenu1 = mymenu1
         self.mymenu2 = mymenu2
         self.mymenu3 = mymenu3
         self.mymenu4 = mymenu4
         self.mymenu5 = mymenu5
         self.mymenu6 = mymenu6
         self.mymenu7 = mymenu7
         self.mymenu8 = mymenu8
         self.mymenu9 = mymenu9
         self.mymenu10 = mymenu10

     def AllOptionsCallBack(self):
        base_name = self.mymenu1.var.get()
        log_name = self.mymenu2.get()
        num_batches_per_run = self.mymenu3.get()
        total_runs = self.mymenu4.get()
        num_img_per_folder = self.mymenu5.get()
        deg_type = self.mymenu6.var.get()
        initial_deg = self.mymenu7.get()
        steps = self.mymenu8.get()
        step_size = self.mymenu9.get()
        topN = self.mymenu10.get()

        print base_name

        #self, croot, iroot, logName, modelName, startingBatch, numOfBatch, numPerBatch, dataSuffix='', debug=False
        #self, typ, init, step, inc, topN, nthrun, thresh=0.5
   
        for i in range(int(total_runs):
            tester = testmodule.CaffeTester(croot, iroot, 
                base_name, '', i, int(num_batches_per_run), int(num_img_per_folder), log_name)
            tester.degrade(deg_type, int(initial_deg), int(steps), 
                int(step_size), int(topN), i, 0.5)

Dragonfly = tk.Tk()
Dragonfly.geometry('900x800+400+300')
Dragonfly.title('Dragonfly')
Dragonfly.columnconfigure(0, weight=1)
Dragonfly.columnconfigure(1, weight=1)  
Dragonfly.columnconfigure(2, weight=1)  
mainlabel = tk.Label(text='Model Parameters', font=('calibri', (14)),
                  fg='dark blue').grid(row=0, column=0, columnspan=3, pady=20)


Dragonfly.columnconfigure(3, weight=1)  
Dragonfly.columnconfigure(4, weight=1)  
Dragonfly.columnconfigure(5, weight=1)  
mainlabel = tk.Label(text='Run Parameters', font=('calibri', (14)),
                  fg='dark blue').grid(row=4, column=0, columnspan=3, pady=20)


Dragonfly.columnconfigure(6, weight=1)  
Dragonfly.columnconfigure(7, weight=1)
Dragonfly.columnconfigure(8, weight=1)  
Dragonfly.columnconfigure(9, weight=1)  
Dragonfly.columnconfigure(10, weight=1)  
mainlabel = tk.Label(text='Degradation Parameters', font=('calibri', (14)),
                  fg='dark blue').grid(row=8, column=0, columnspan=3, pady=20)


##

mymenu1 = MyOptionMenu(Dragonfly, 'Select base model', 'googlenet', 'alexnet', 'ilsvrc13')
mymenu1.grid(row=2, column=0, pady=10, padx=10,)
# b1_1 = tk.Button(Dragonfly, text="Run", fg='blue', command=mymenu1.callback)
# b1_1.grid(row=3, column=0, pady=10, padx=10,)

mymenu2 = tk.Entry(Dragonfly)
mymenu2.grid(row=2, column=1, pady=10, padx=10,)
# b2_2 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data2)
# b2_2.grid(row=3, column=1, pady=10, padx=10,)


##

mymenu3 = tk.Entry(Dragonfly)
mymenu3.grid(row=6, column=0, pady=10, padx=10,)
# b3_3 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data3)
# b3_3.grid(row=7, column=0, pady=10, padx=10,)

mymenu4 = tk.Entry(Dragonfly)
mymenu4.grid(row=6, column=1, pady=10, padx=10,)
# b4_4 = tk.Button(Dragonfly,text='Enter',command=submit_data4)
# b4_4.grid(row=7, column=1, pady=10, padx=10,)

mymenu5 = tk.Entry(Dragonfly)
mymenu5.grid(row=6, column=2, pady=10, padx=10,)
# b5_5 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data5)
# b5_5.grid(row=7, column=2, pady=10, padx=10,)

##

mymenu6 = MyOptionMenu(Dragonfly, 'Select deg type', ['clean'], ['gaussian'], ['quality'], ['resize'])
mymenu6.grid(row=10, column=0, pady=10, padx=10,)
# b6_6 = tk.Button(Dragonfly, text="Run", fg='blue', command=mymenu6.callback)
# b6_6.grid(row=11, column=0, pady=10, padx=10,)

mymenu7 = tk.Entry(Dragonfly)
mymenu7.grid(row=10, column=1, pady=10, padx=10,)
# b7_7 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data7)
# b7_7.grid(row=11, column=1, pady=10, padx=10,)

mymenu8 = tk.Entry(Dragonfly)
mymenu8.grid(row=10, column=2, pady=10, padx=10,)
# b8_8 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data8)
# b8_8.grid(row=11, column=2, pady=10, padx=10,)

mymenu9 = tk.Entry(Dragonfly)
mymenu9.grid(row=10, column=3, pady=10, padx=10,)
# b9_9 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data9)
# b9_9.grid(row=11, column=3, pady=10, padx=10,)

mymenu10 = tk.Entry(Dragonfly)
mymenu10.grid(row=13, column=0, pady=10, padx=10,)
# b10_10 = tk.Button(Dragonfly, text="Run", fg='blue', command=submit_data10)
# b10_10.grid(row=14, column=0, pady=10, padx=10,)


m1label = tk.Label(text='Base Model', font=('calibri', (12)),
                fg='dark green').grid(row=1, column=0, pady=10, padx=10,)
m1labe2 = tk.Label(text='Log Name (name of log file to output the test results)', font=('calibri', (12)),
                fg='dark green').grid(row=1, column=1, pady=10, padx=10,)
m1labe4 = tk.Label(text='Number of Batches per Run', font=('calibri', (12)),
                fg='dark green').grid(row=5, column=0, pady=10, padx=10,)
m1labe4 = tk.Label(text='Total Runs', font=('calibri', (12)),
                fg='dark green').grid(row=5, column=1, pady=10, padx=10,)
m1labe5 = tk.Label(text='Number of Images in Batch Folder', font=('calibri', (12)),
                fg='dark green').grid(row=5, column=2, pady=10, padx=10,)
m1labe6 = tk.Label(text='Degradation Type', font=('calibri', (12)),
                fg='dark green').grid(row=9, column=0, pady=10, padx=10,)
m1labe7 = tk.Label(text='Initial Degradation', font=('calibri', (12)),
                fg='dark green').grid(row=9, column=1, pady=10, padx=10,)
m1labe8 = tk.Label(text='Number of Steps', font=('calibri', (12)),
                fg='dark green').grid(row=9, column=2, pady=10, padx=10,)
m1labe9 = tk.Label(text='Step Size', font=('calibri', (12)),
                fg='dark green').grid(row=9, column=3, pady=10, padx=10,)
m1labe10 = tk.Label(text='Top N (Top N values to select)', font=('calibri', (12)),
                fg='dark green').grid(row=12, column=0, pady=10, padx=10,)



allOptionsObject = AllOptions(mymenu1, mymenu2, mymenu3, mymenu4, mymenu5, mymenu6, mymenu7, mymenu8, mymenu9,mymenu10)
Submit = tk.Button(Dragonfly, text="Submit", fg='blue', command=allOptionsObject.AllOptionsCallBack)
Submit.grid(row=15, column=1, pady=10, padx=10,)


Dragonfly.mainloop()