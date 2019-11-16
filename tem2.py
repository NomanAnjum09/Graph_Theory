from tkinter import *

# pip install pillow
from PIL import Image, ImageTk

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=0)
        
        load = Image.open("/root/Pictures/Graph3.png")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

root=Tk() 
root = Toplevel()
  
app = Window(root)
root.wm_title("Tkinter window")
root.geometry("1920x1080")
root.mainloop()