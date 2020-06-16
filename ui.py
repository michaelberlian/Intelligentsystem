import tkinter
from tkinter import filedialog as tkFileDialog
import os

#to choose file
def choosefile():
    root = tkinter.Tk()
    root.withdraw()

    currdir = os.getcwd()
    tempdir = tkFileDialog.askopenfilename(initialdir = "{}/Images".format(currdir) ,title = "Select file",filetypes = (("all files","*.jpg"),("all files","*.jpeg"),("all files","*.PNG")))
    if len(tempdir) > 0:
        print ("You chose %s" % tempdir)
    
    return tempdir
