import joblib, os
import numpy as np
import ypoften as of

cwd = os.path.join("") # replace with your own folder path
cvmodel = "vgg16hybrid"

imgnamefile = os.path.join(cwd,"imgname.txt")
imgnamefile = open(imgnamefile, "r")
exfolder = os.path.join(cwd,"img exfeature", "")
features_array = []

for lineindex, line in enumerate(imgnamefile.readlines()[:]):
    imgname = line.rstrip('\r\n')
    ex_feature_path = os.path.join(cwd, 'img exfeature', cvmodel, imgname + ".dat")
    imgexfeatures = joblib.load(ex_feature_path)
    flatten_features = np.ndarray.flatten(imgexfeatures)
    features_array.append(flatten_features)

features_array = np.array(features_array)
features_savepath = os.path.join(cwd, 'img exfeature','features combine', cvmodel + ".dat")
of.create_path(features_savepath)
joblib.dump(features_array,features_savepath)

print("DONE"*20)