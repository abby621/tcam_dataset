import glob
import os
import cv2
import numpy as np
import random
from PIL import Image

testDir = './images/test/'
testFiles = glob.glob(os.path.join(testDir,'*/*.jpg'))

peopleDir = './images/people_crops/test/'
peopleMaskFiles = glob.glob(os.path.join(peopleDir,'*.png'))

min_occlusion = [5, 25, 45, 65]
max_occlusion = [25, 45, 65, 85]

outBaseDir = './images/occluded_test/'

def getMaskedImage(image_file,mask_file,min_val,max_val):
    img = cv2.imread(image_file)
    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    scale = np.random.randint(min_val,max_val)/float(100)
    resized_mask = cv2.resize(mask,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    top = np.random.randint(0,img.shape[0]-resized_mask.shape[0])
    left = np.random.randint(0,img.shape[1]-resized_mask.shape[1])
    new_mask = np.ones((img.shape[0],img.shape[1]))*255.0
    new_mask[top:top+resized_mask.shape[0],left:left+resized_mask.shape[1]] = resized_mask
    new_mask[new_mask<255] = 0
    new_mask /= 255
    new_mask_col = np.tile(np.expand_dims(new_mask,2),(1,1,img.shape[2]))
    new_img = img * new_mask_col
    return new_img

for difficultyLevel in range(0,3):
    diffDir = os.path.join(outBaseDir,str(difficultyLevel))
    if not os.path.exists(diffDir):
        os.makedirs(diffDir)
    for f in testFiles:
        img = None
        cls = f.split('/')[3]
        outDir = os.path.join(diffDir,cls)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        img = getMaskedImage(f,random.choice(peopleMaskFiles),min_occlusion[difficultyLevel],max_occlusion[difficultyLevel])
        tries = 0
        while img is None and tries < 5:
            img = getMaskedImage(f,random.choice(peopleMaskFiles),min_occlusion[difficultyLevel],max_occlusion[difficultyLevel])
            tries += 1
        try:
            pilImg = Image.fromarray(img.astype('uint8'))
            b, g, r = pilImg.split()
            pilImg2 = Image.merge("RGB", (r, g, b))
            outFile = os.path.join(outDir,f.split('/')[-1])
            pilImg2.save(outFile)
            print 'Saved: ', outFile
        except:
            print 'Bad image: ', f
