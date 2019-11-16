#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:27:46 2019

@author: root
"""

from tkinter import *
from tkinter.ttk import *
import numpy as np
from tkinter.font import Font
import os
from PIL import ImageTk,Image



root = Tk()
#canvas=Canvas(root,width=1200,height=900)
#image=ImageTk,PhotoImage(Image.open('/root/Pictures/Graph1.png'))
#canvas.create_image(0,0,anchor=NW,image=image)
#canvas.pack()




root.geometry('1200x900') 
root.title("ProjectAlogrithmCS302")

files = iter(np.arange(1,3000))
downloaded = IntVar()





def loading():
    try:
        downloaded.set(next(files)) # update the progress bar
        root.after(1, loading) # call this function again in 1 millisecond
    except StopIteration:
        # the files iterator is exhausted
        root.destroy()
        import SelectFile.py        
        






my_font = Font(family="Times New Roman", size=40, weight="bold" )
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




progress= Progressbar(root, orient = 'horizontal', maximum = 3000, variable=downloaded, mode = 'determinate')
progress.pack(fill=BOTH,side = BOTTOM)
loading()
#start = ttk.Button(root,text='start',command=loading)
#start.pack(pady = 20,side = BOTTOM)

root.mainloop()

