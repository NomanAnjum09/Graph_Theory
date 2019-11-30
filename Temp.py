from tkinter import *
from PIL import Image, ImageTk
from tkinter.ttk import *
import numpy as np
from tkinter.font import Font
import os


root = Toplevel()
root.title("Title")
root.geometry("1920x1080")
root.configure(background="black")

files = iter(np.arange(1,3000))
downloaded = IntVar()




def loading():
    try:
        downloaded.set(next(files)) # update the progress bar
        root.after(1, loading) # call this function again in 1 millisecond
    except StopIteration:
        # the files iterator is exhausted
        root.destroy()
        #import SelectFile.py 


class Example(Frame):
    
    def __init__(self, master, *pargs):
        Frame.__init__(self, master, *pargs)
        self.image = Image.open("kruskal.png")
        self.img_copy= self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background = Label(self, image=self.background_image)
        self.background.pack(fill=BOTH, expand=YES)
        self.background.bind('<Configure>', self._resize_image)
        my_font = Font(family="Times New Roman", size=35, weight="bold" )
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="Design And Analysis Of Alogorithm" ,font=my_font).pack()
        Label(root,  text="Shortest Path Algorithm Implementation" ,font=("Helvetica",35)).pack()
        Label(root,  text="Group Members:" ,font=("Helvetica",30)).pack()
        Label(root,  text="Noman Anjum           17K-3755" ,font=("Helvetica",25)).pack()
        Label(root,  text="Muhammad Ahsan Siddiq 17K-3650" ,font=("Helvetica",25)).pack()
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="" ,font=my_font).pack()
        Label(root,  text="Please Wait While Loading ..." ,font=my_font).pack()



    def _resize_image(self,event):
        new_width = event.width
        new_height = event.height
        self.image = self.img_copy.resize((new_width, new_height))
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background.configure(image =  self.background_image)


progress= ttk.Progressbar(root, orient = 'horizontal', maximum = 3000, variable=downloaded, mode = 'determinate')
progress.pack(fill=BOTH,side = BOTTOM)
loading()
e = Example(root)
e.pack(fill=BOTH, expand=YES)
root.mainloop()