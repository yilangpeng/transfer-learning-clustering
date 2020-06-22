import joblib, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ypoften as of

cwd = os.path.join("") # replace with your own folder path
cvmodel = "vgg16hybrid"

imgnamefile = os.path.join(cwd,"imgname.txt")
imgnamefile = open(imgnamefile, "r")
exfolder = os.path.join(cwd,"img exfeature", "")

features_savepath = os.path.join(cwd,'img exfeature','features combine',cvmodel+'.dat')

features_array = joblib.load(features_savepath)
x = pd.DataFrame(features_array)

# standardizing the features
x = StandardScaler().fit_transform(x)

# PCA
pca = PCA(0.95) #95% of the variance is retained.
pca.fit(x)
x_pca = pca.transform(x)

# save PCA
savefolder = os.path.join(cwd,cwd,'img exfeature','features PCA')
pca_savepath = os.path.join(savefolder, cvmodel+'.dat')
of.create_path(pca_savepath)
joblib.dump(x_pca, pca_savepath)

# save components
components = pca.components_
savepath = os.path.join(savefolder, cvmodel + 'components.txt')
components = pd.DataFrame(components)
components.to_csv(savepath, header=None, index=None, sep='\t', mode='a')

# save variance explained ratio
variance = pca.explained_variance_ratio_
savepath = os.path.join(savefolder, cvmodel + ' variance.txt')
variance = pd.DataFrame(variance)
variance.to_csv(savepath, header=None, index=None, sep='\t', mode='a')

print("DONE"*20)
