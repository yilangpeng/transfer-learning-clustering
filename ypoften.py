# -*- coding: utf-8 -*-
import os, shutil

def create_path(filepath):
    filepathfolder = os.path.dirname(filepath) 
    if not os.path.exists(filepathfolder): os.makedirs(filepathfolder)

def copy_file(filepath1, filepath2):
    filepathfolder2 = os.path.dirname(filepath2) 
    if not os.path.exists(filepathfolder2): os.makedirs(filepathfolder2)
    shutil.copy2(filepath1, filepath2)
