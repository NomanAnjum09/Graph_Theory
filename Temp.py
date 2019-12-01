#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:00:46 2019

@author: root
"""
import sys
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from tkinter.font import Font
from os import listdir
from PIL import Image as PImage
import numpy as np


G=nx.DiGraph()
GP=nx.DiGraph()
GK=nx.DiGraph()
GB=nx.DiGraph()
GD=nx.DiGraph()
GF=nx.DiGraph()
#Graph Plotter Via Networkx and PyPLot
def plotter(Graph,name):
    pos=nx.get_node_attributes(Graph,'pos')
    fig, ax = plt.subplots(figsize=(40, 30),dpi=100)
    nx.draw_networkx_nodes(Graph,pos,with_labels=True,ax=ax)
    labels = nx.get_edge_attributes(Graph,'weight')
    nx.draw_networkx_labels(Graph,pos)
    nx.draw_networkx_edge_labels(Graph,pos,edge_labels=labels)
    nx.draw_networkx_edges(Graph,pos,edge_labels=labels)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    if(name=='prims'):
        path=('prims.png')
    elif(name=='kruskal'):
        path=('kruskal.png')
    elif(name=='dijsktra'):
        path=('dijsktra.png')
    elif(name=='bellman'):
        path=('bellmanFord.png')
    elif(name=='floyd'):
        path=('FloydWarshall.png')
    plt.savefig(path)
    img = PImage.open(path)
    img.show()
    return





#File Opener And Parser
#Insert All vertice and and edges in graph and generate INuputted Graph
def open_file():
    fileopen = askopenfile(mode ='r', filetypes =[('Text Files', '*.txt')])
    if fileopen is not None:        
        G.clear()
        GP.clear
        GK.clear()
        GB.clear()
        GD.clear()
        extra=fileopen.readline()
        extra=fileopen.readline()
        global no_of_nodes
        no_of_nodes=fileopen.readline()
        no_of_nodes=int(no_of_nodes)
        print(no_of_nodes)
        extra=fileopen.readline()
        global for_node
        global for_x_axis
        global for_y_axis
        
        
        for_node=[]
        for_x_axis=[]
        for_y_axis=[]
        for loop in range(no_of_nodes):
            node_info=fileopen.readline()
            node,x_axis,y_axis=node_info.split('\t')
            y_axis=float(y_axis)
            x_axis=float(x_axis)
            node=int(node)
            #print(node)
            for_node.append(node)
            #print(for_node)
            for_y_axis.append(y_axis)
            #print(y_axis)
            for_x_axis.append(x_axis)
            #print(x_axis)
            G.add_node(node,pos=(x_axis,y_axis))

        extra=fileopen.readline()
        NodeNo1=[]
        NodeNo2=[]
        bandwith=[]
        countfornode1=0
        for outerloop in range(no_of_nodes):
            node_info=fileopen.readline()
            info=node_info.split('\t')
            info_length=len(info)
            for innerloop in range(1,(info_length-4),4):
                NodeNo1.append(countfornode1)
                forNode2=info[innerloop]
                forNode2=int(forNode2)
                NodeNo2.append(forNode2)
                forbandwith=info[innerloop+2]
                forbandwith=float(forbandwith)
                forbandwith=forbandwith/10000000
                bandwith.append(forbandwith)
            countfornode1=countfornode1+1
        for i in range (len(NodeNo1)):
            G.add_edge(NodeNo1[i],NodeNo2[i],weight=bandwith[i])
        lengthofnode=len(NodeNo1)
        global graph1
        graph1=[[0 for x in range(no_of_nodes)] for y in range(no_of_nodes)]
        print(lengthofnode)
        #for kruskal definition
        global gk
        global g
        gk = KruskalGraph(no_of_nodes)
        g=Graph(no_of_nodes)
        for i in range(lengthofnode):
            graph1[NodeNo1[i]][NodeNo2[i]]=bandwith[i]
            g.addEdge(NodeNo1[i],NodeNo2[i],bandwith[i])
            gk.addEdge(NodeNo1[i],NodeNo2[i],bandwith[i])
        extra=fileopen.readline()
        global source
        source=fileopen.readline()
        source=int(source,10)
        #print(graph1)











#Floyd Warshall Tree IMplementation
class FloydWarshall():
    def __init__(self,Graph):
        self.graph=Graph
        self.v = no_of_nodes
        self.p = [[0 for i in range(no_of_nodes)] for j in range(no_of_nodes)]
    
    def print_path(self):
        total=0
        f=open("Floyd Warshall.txt","w+")
        f.write("Floyd Warshall Shortest Path Algorithm\n")
        for i in range(no_of_nodes):
            n=for_node[i]
            x=for_x_axis[i]
            y=for_y_axis[i]
            GF.add_node(n,pos=(x,y))
        print ("Vertex Distance from Source")
        f.write("Vertex Distance from Source\n")
        for i in range(self.v):
            print(self.p[source][i],i,self.graph[source][i])
            f.write(str(self.p[source][i])+"-->"+str(i)+"  "+str(self.graph[source][i])+'\n')
            GF.add_edge(self.p[source][i],i,weight=self.graph[source][i])
            total=total+self.graph[source][i]
        print("Total cost: ",total)
        f.write("Total Cost: "+str(total)+'\n')
        plotter(GF,'floyd')
        f.close()
    def floyd_warshal(self):
        for i in range(0,self.v):
            for j in range(0,self.v):
                self.p[i][j]=0
                
        for i in range(0,self.v):
            for j in range(0,self.v):
                self.p[i][j] = i
                if (i != j and self.graph[i][j] == 0): 
                    self.p[i][j] = -30000 
                    self.graph[i][j] = 30000 # set zeros to any large number which is bigger then the longest way
                    
        for k in range(0,self.v):
            for i in range(0,self.v):
                for j in range(0,self.v):
                    if self.graph[i][j] > self.graph[i][k] + self.graph[k][j]:
                        self.graph[i][j] = self.graph[i][k] + self.graph[k][j]
                        self.p[i][j] = self.p[k][j]
        self.print_path()
    

            
        
      






#Dijsktra Implementation


class DjikstraGraph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist,source,total):
        f=open("Dijsktra.txt","w+")
        f.write("Dijsktra Shortest Path Algorithm\n")
        total=0
        for i in range(no_of_nodes):
            n=for_node[i]
            x=for_x_axis[i]
            y=for_y_axis[i]
            GD.add_node(n,pos=(x,y))
        print ("Vertex Distance from Source")
        f.write("Vertex Distance from Source\n")
        for node in range(self.V):
            #print(source[node])
            #print(self.graph[node][source[node]])
            total = total + dist[node]
            print (source[node], node, dist[node])
            f.write(str(source[node][0])+"-->"+str(node)+"  "+str(dist[node])+'\n')
            GD.add_edge(source[node][0],node,weight=dist[node])
        print("Total cost: ",total)
        f.write("Total Cost: "+str(total)+'\n')
        plotter(GD,'dijsktra')
        f.close()
        #print(total)
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        source1 = [[0 for i in range(2)] for j in range(self.V)]
        sptSet = [False] * self.V
        total=0

        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                    source1[v][0]=u
                    source1[v][1]=v
                    dist[v] = dist[u] + self.graph[u][v]
                    total=total+dist[v]
        self.printSolution(dist,source1,total)

#Dijsktra Finished











#Bellman Ford Graph Implemntaion

class Graph:

    def __init__(self, vertices):
        self.V = vertices # No. of vertices
        self.graph = [] # default dictionary to store graph
        


    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def printArr(self, dist,total,source1):
        f=open("BellmanFord.txt","w+")
        f.write("BellmanFord Shortest Path Algorithm\n")
        total=0
        for i in range(no_of_nodes):
            n=for_node[i]
            x=for_x_axis[i]
            y=for_y_axis[i]
            GB.add_node(n,pos=(x,y))
        print("Vertex Distance from Source")
        for i in range(self.V):
            print(source1[i],i, dist[i])
            GB.add_edge(source1[i][0],i,weight=dist[i])
            f.write(str(source1[i][0])+"-->"+str(i)+"  "+str(dist[i])+"\n")
            total=total+dist[i]
        print(total)
        f.write("Total Cost: "+str(total))
        plotter(GB,'bellman')

    def BellmanFord(self, src):
        dist = [float("Inf")] * self.V
        dist[src] = 0
        total = 0
        source1 = [[0 for i in range(2)] for j in range(self.V)]


        for i in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        source1[v][0]=u
                        source1[v][1]=v
                        total = total + dist[v]

        for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        print ("Graph contains negative weight cycle")
                        return

        self.printArr(dist,total,source1)

#Bellman Ford Finshed









#Kruskal Graph Implementation
class KruskalGraph:

    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = [] # default dictionary
                                # to store graph


    # function to add an edge to graph
    def addEdge(self,u,v,w):
        self.graph.append([u,v,w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):

        result =[] #This will store the resultant MST

        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for result[]
        total = 0
            # Step 1:  Sort all the edges in non-decreasing
                # order of their
                # weight.  If we are not allowed to change the
                # given graph, we can create a copy of graph
        self.graph =  sorted(self.graph,key=lambda item: item[2])

        parent = [] ; rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        # Number of edges to be taken is equal to V-1
        while e < self.V -1 :

            # Step 2: Pick the smallest edge and increment
                    # the index for next iteration
            u,v,w =  self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent ,v)

            # If including this edge does't cause cycle,
                        # include it in result and increment the index
                        # of result for next edge
            if x != y:
                e = e + 1
                result.append([u,v,w])
                self.union(parent, rank, x, y)
            # Else discard the edge
                
        # print the contents of result[] to display the built MST]
        f=open("Kruskal.txt","w+")
        f.write("Kruskal Minimum Spanning Tree\n\n")
        for i in range(no_of_nodes):
            n=for_node[i]
            x=for_x_axis[i]
            y=for_y_axis[i]
            GK.add_node(n,pos=(x,y))
        print ("Following are the edges in the constructed MST")
        f.write("Following are the edges in the constructed MST\n\n")
        for u,v,weight  in result:
            print ("%d -- %d == %f" % (u,v,weight))
            f.write(str(u)+"-->"+str(v)+"  "+str(weight)+"\n")
            total = total + weight
            GK.add_edge(u,v,weight=weight)
        print(total)
        f.write("Total Cost: "+str(total))
        plotter(GK,'kruskal')
        return

#Krsukal Finished








#Prims Graph Implementation

class PrimsGraph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                    for row in range(vertices)]

    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent):
        GP.clear()
        f=open("Prims.txt","w+")
        f.write("Prims Minimum Spanning Tree\n\n")
        total = 0
        for i in range(no_of_nodes):
            n=for_node[i]
            x=for_x_axis[i]
            y=for_y_axis[i]
            GP.add_node(n,pos=(x,y))
        print ("Edge \t Weight")
        f.write("Edge \t Weight\n")
        for i in range(1, self.V):
            total = total + self.graph[i][parent[i]]
            print (parent[i], "to", i, "\t", self.graph[i][ parent[i]])
            GP.add_edge(parent[i],i,weight=self.graph[i][ parent[i]])
            f.write(str(parent[i])+"-->"+str(i)+"  "+str(self.graph[i][ parent[i]])+"\n")
        print(total)
        f.write("Total Cost: "+str(total))
        plotter(GP,'prims')
#        pos=nx.get_node_attributes(GP,'pos')
#        fig, ax = plt.subplots(figsize=(40, 30),dpi=100)
#        nx.draw_networkx_nodes(GP,pos,with_labels=True,ax=ax)
#        labels = nx.get_edge_attributes(GP,'weight')
#        nx.draw_networkx_labels(GP,pos)
#        nx.draw_networkx_edge_labels(GP,pos,edge_labels=labels)
#        nx.draw_networkx_edges(GP,pos,edge_labels=labels)
#        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#        plt.savefig('/root/Pictures/Prims.png')
        return
        
        
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initilaize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        total=0
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u

        self.printMST(parent)

#Prims Finished






#Graph Caller Segment
        
def Floyd_Warshall():
    print("Floyd Warshall Algorithm")
    gf=FloydWarshall(graph1)
    gf.floyd_warshal()    

def bellman_ford():
    print("BellmanFord SP")
    g.BellmanFord(source)

def kruskal():
    print("Kruskal MST")
    gk.KruskalMST()

def dijsktra():
    print("Dijsktra SP")
    gr = DjikstraGraph(no_of_nodes)
    gr.graph=graph1
    gr.dijkstra(source)

def prims():
    print("PRIMS MST")
    gr = PrimsGraph(no_of_nodes)
    gr.graph = graph1
    gr.primMST()


def plot_actual():
    pos=nx.get_node_attributes(G,'pos')
    fig, ax = plt.subplots(figsize=(40, 30),dpi=100)
    nx.draw_networkx_nodes(G, pos,with_labels=True,ax=ax)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    nx.draw_networkx_edges(G,pos,edge_labels=labels)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    path=('ActualGraph.png')
    plt.savefig(path)
    img = PImage.open(path)
    img.show()
    return




        


root = Tk()
root.geometry('1200x900')

my_font = Font(family="Times New Roman", size=40, weight="bold" )
Label(root,  text="" ,font=my_font).pack()
Label(root,  text="Select A .txt File From Your System" ,font=my_font).pack()
btn1 = Button(root, text ='Open', command = lambda:open_file())
btn1.pack(pady = 20)
btn2 = Button(root, text ='SeeActualGraph',command = lambda:plot_actual())
btn2.pack(pady = 20)
Label(root,  text="Minimum Spanning Trees Algorithm:" ,font=my_font).pack()

btn3 = Button(root, text ='Run Prims',command = lambda:prims())
btn3.pack(pady = 20)
btn4 = Button(root, text ='Run Kruskal',command = lambda:kruskal())
btn4.pack(pady = 10)
Label(root,  text="" ,font=my_font).pack()
Label(root,  text="Shortest Path Algorithm:" ,font=my_font).pack()
btn5 = Button(root, text ='Run Dijsktra',command = lambda:dijsktra())
btn5.pack(pady = 20)
btn8 = Button(root, text ='Run FloydWarshall',command = lambda:Floyd_Warshall())
btn8.pack(pady = 20)
btn6 = Button(root, text ='Run BellmanFord',command = lambda:bellman_ford())
btn6.pack(pady = 10)
Label(root,  text="" ,font=my_font).pack()
Label(root,  text="Clustering Cofficient(Local Clustering):" ,font=my_font).pack()
btn7 = Button(root, text ='Run Local Clustering')
btn7.pack(pady = 10)




mainloop()