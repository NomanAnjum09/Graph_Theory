import networkx as nx
import matplotlib.pyplot as plt
from tkinter import *  
from PIL import ImageTk,Image

root=Tk()
#canvas = Canvas(root,  width=400, height=400)
#canvas.pack()
#img = PhotoImage(file="/root/Pictures/pic22.png")
#canvas.create_image(10, 10, anchor=NW, image=img)

G=nx.DiGraph()
i=1
G.add_node(i,pos=(i,i))
G.add_node(2,pos=(2,3))
G.add_node(3,pos=(1,0))
G.add_edge(1,2,weight=0.5)
G.add_edge(1,3,weight=9.8)

pos=nx.get_node_attributes(G,'pos')
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos,with_labels=True,ax=ax)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_labels(G,pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
nx.draw_networkx_edges(G,pos,edge_labels=labels)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.savefig('/root/Pictures/pic22.png')
root.mainloop() 