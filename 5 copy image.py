import joblib, os, random
import numpy as np
import pandas as pd
import ypoften as of

cwd = os.path.join("") # replace with your own folder path
cvmodel = "vgg16hybrid"
clmethod = "KMeans"

savefolder = "vgg16hybrid KMeans PCA200"

def save_img(random_id, imgname, label, K):
    imgpath1=os.path.join(cwd,'img all',imgname)
    newimgname = str(random_id)+' '+imgname
    imgpath2 = os.path.join(cwd,'img cluster',savefolder, str(K),str(label),newimgname)
    of.copy_file(imgpath1, imgpath2)

for K in range(5, 21):
    print('number of cluster',K)
    filepath = os.path.join(cwd,'img cluster',savefolder,str(K),'label.txt')
    df = pd.read_csv(filepath, sep ='\t', header = None)
    df.columns = ['imgname',"label"]

    dr = df.sample(frac=1, random_state=42) # random shuffle images
    dr['random_id'] = np.arange(len(dr)) # add random id

    for labelK in range(0, K): 
        dc = dr.loc[dr['label'] == labelK] # select images from each cluster
        dc = dc.iloc[:20,:] # select 20 images from each cluster
        dc.apply(lambda row: save_img(row['random_id'],row['imgname'],row['label'], K),axis=1)

    print("DONE"*20)
print("DONE"*20)
