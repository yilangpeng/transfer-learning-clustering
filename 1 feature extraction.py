import os, joblib
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import ypoften as of

cvmodel = "vgg16"
cvmodel = "vgg16hybrid"

if cvmodel == "vgg16":
    from keras.applications.vgg16 import VGG16
    base_model = VGG16(weights="imagenet", include_top=True)
    print(base_model.summary())
    feature_model = Model(base_model.input, base_model.get_layer('fc1').output)
    img_size = (224, 224)
    
elif cvmodel == "vgg16hybrid":
    from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
    base_model = VGG16_Hybrid_1365(weights='places', include_top=True, input_shape=(224, 224, 3))
    print(base_model.summary())
    feature_model = Model(base_model.input, base_model.get_layer('fc2').output)
    img_size = (224, 224)

print(feature_model.summary())

cwd = os.path.join("") # replace with your own folder path
imgnamefile = os.path.join(cwd,"imgname.txt")
imgnamefile = open(imgnamefile, "r")
exfolder = os.path.join(cwd,"img exfeature", "")

for j, line in enumerate(imgnamefile.readlines()[:]):
    imgname = line.rstrip('\n\r')
    print(j, imgname)

    imgpath = os.path.join(cwd, 'img all', imgname)
    img = image.load_img(imgpath, target_size=img_size) # read the image 
    image_array = image.img_to_array(img) # convert the image to a numpy array

    image_expand = np.expand_dims(image_array, 0)
    x_train = preprocess_input(image_expand) # normalize the image to 0-1 range
    features_x = feature_model.predict(x_train) # extract features for each image

    featuresavepath = os.path.join(exfolder, cvmodel, imgname+".dat")
    of.create_path(featuresavepath)
    joblib.dump(features_x[0], featuresavepath) # save the extracted features
    print("==SUCCESS=="*20)

print("DONE"*20)
