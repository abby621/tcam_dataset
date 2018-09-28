# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import numpy as np
import cv2
import random
import os
import glob
import socket

peopleDir = './images/people_crops/train/'

class CombinatorialTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        # this is SUPER hacky -- if the test file is 'occluded' then the class is in the 5th position, not the 4th
        if 'occluded' in image_list:
            clsPos = 4
        else:
            clsPos = 3

        self.hotels = {}
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        for line in f:
            temp = line.strip('\n').split(' ')
            hotel = int(temp[0].split('/')[clsPos])
            self.hotels[hotel] = {}
            self.hotels[hotel]['ims'] = temp
        for hotel in self.hotels.keys():
            if len(self.hotels[hotel]['ims']) < self.numPos:
                self.hotels.pop(hotel)
            else:
                self.hotels[hotel]['sources'] = np.array([im.split('/')[clsPos+1] for im in self.hotels[hotel]['ims']])

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*.png'))

    def getBatch(self):
        numClasses = self.batchSize/self.numPos # need to handle the case where we need more classes than we have?
        classes = np.random.choice(self.hotels.keys(),numClasses,replace=False)

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])

        labels = np.zeros([self.batchSize],dtype='int')
        ims = []

        ctr = 0
        for cls in classes:
            cls = int(cls)
            clsPaths = self.hotels[cls]['ims']
            clsSources = self.hotels[cls]['sources']
            tcamInds = np.where(clsSources=='tcam')[0]
            exInds = np.where(clsSources=='expedia')[0]
            if len(tcamInds) >= self.numPos/2 and len(exInds) >= self.numPos/2:
                numTcam = self.numPos/2
                numEx = self.numPos - numTcam
            elif len(tcamInds) >= self.numPos/2 and len(exInds) < self.numPos/2:
                numEx = len(exInds)
                numTcam = self.numPos - numEx
            else:
                numTcam = len(tcamInds)
                numEx = self.numPos - numTcam

            random.shuffle(tcamInds)
            random.shuffle(exInds)

            for j1 in np.arange(numTcam):
                imPath = self.hotels[cls]['ims'][tcamInds[j1]]
                img = self.getProcessedImage(imPath)
                if img is not None:
                    batch[ctr,:,:,:] = img
                labels[ctr] = cls
                ims.append(imPath)
                ctr += 1

            for j2 in np.arange(numEx):
                imPath = self.hotels[cls]['ims'][exInds[j2]]
                img = self.getProcessedImage(imPath)
                if img is not None:
                    batch[ctr,:,:,:] = img
                labels[ctr] = cls
                ims.append(imPath)
                ctr += 1

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            img = cv2.flip(img,1)

        # if self.isTraining:
        #     img = doctor_im(img)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        img = img - self.meanImage

        return img

    def getPeopleMasks(self):
        which_inds = random.sample(np.arange(0,len(self.people_crop_files)),self.batchSize)

        people_crops = np.zeros([self.batchSize,self.crop_size[0],self.crop_size[1]])
        for ix in range(0,self.batchSize):
            people_crops[ix,:,:] = self.getImageAsMask(self.people_crop_files[which_inds[ix]])

        people_crops = np.expand_dims(people_crops, axis=3)

        return people_crops

    def getImageAsMask(self, image_file):
        img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # how much of the image should the mask take up
        scale = np.random.randint(30,70)/float(100)
        resized_img = cv2.resize(img,(int(self.crop_size[0]*scale),int(self.crop_size[1]*scale)))

        # where should we put the mask?
        top = np.random.randint(0,self.crop_size[0]-resized_img.shape[0])
        left = np.random.randint(0,self.crop_size[1]-resized_img.shape[1])

        new_img = np.ones((self.crop_size[0],self.crop_size[1]))*255.0
        new_img[top:top+resized_img.shape[0],left:left+resized_img.shape[1]] = resized_img

        new_img[new_img<255] = 0
        new_img[new_img>1] = 1

        return new_img

class NonTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        # this is SUPER hacky -- if the test file is 'occluded' then the class is in the 5th position, not the 4th
        if 'occluded' in image_list:
            clsPos = 4
        else:
            clsPos = 3

        self.hotels = {}
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        for line in f:
            temp = line.strip('\n').split(' ')
            hotel = int(temp[0].split('/')[clsPos])
            self.hotels[hotel] = {}
            self.hotels[hotel]['ims'] = temp
        for hotel in self.hotels.keys():
            self.hotels[hotel]['sources'] = np.array([im.split('/')[clsPos+1] for im in self.hotels[hotel]['ims']])

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*.png'))

    def getBatch(self):
        batch = np.zeros([self.batch_size, self.crop_size[0], self.crop_size[1], 3])

        inds = np.random.sample(self.indexes)
        ims = self.files[inds]
        classes = self.files[inds]
        for im in ims:
            img = self.getProcessedImage(imPath)
            if img is not None:
                batch[ctr,:,:,:] = img
            ctr += 1

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            img = cv2.flip(img,1)

        # if self.isTraining:
        #     img = doctor_im(img)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        img = img - self.meanImage

        return img

class SameClassSet(CombinatorialTripletSet):
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        # this is SUPER hacky -- if the test file is 'occluded' then the class is in the 5th position, not the 4th
        if 'occluded' in image_list:
            clsPos = 4
        else:
            clsPos = 3

        self.chains = {}
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line.strip('\n').split(' ')
            self.chains[ctr] = {}
            for t in temp:
                hotel = int(t.split('/')[clsPos])
                if not hotel in self.chains[ctr].keys():
                    self.chains[ctr][hotel] = {}
                    self.chains[ctr][hotel]['ims'] = [t]
                else:
                    if t not in self.chains[ctr][hotel]['ims']:
                        self.chains[ctr][hotel]['ims'].append(t)
            for hotel in self.chains[ctr].keys():
                if len(self.chains[ctr][hotel]['ims']) < self.numPos:
                    self.chains[ctr].pop(hotel)
                else:
                    self.chains[ctr][hotel]['sources'] = np.array([im.split('/')[clsPos+1] for im in self.chains[ctr][hotel]['ims']])
            ctr += 1

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*.png'))

    def getBatch(self):
        numClasses = self.batchSize/self.numPos
        chain = np.random.choice(self.chains.keys())

        # half of the classes in the batch should be from the same chain -- this is a version of hard mining,
        # making it so half of the negative examples we see are "harder" because they come from the same chain
        classes = np.zeros(numClasses,dtype='int')
        chains = np.zeros(numClasses,dtype='int')
        while len(self.chains[chain].keys()) < numClasses/2:
            chain = np.random.choice(self.chains.keys())

        classes[:numClasses/2] = np.random.choice(self.chains[chain].keys(),numClasses/2,replace=False)
        chains[:numClasses/2] = chain

        # the other half of the classes should be from random hotels
        for iy in range(numClasses/2,numClasses):
            chain2 = np.random.choice(self.chains.keys())
            while chain2 == chain or len(self.chains[chain2].keys()) < 1:
                chain2 = np.random.choice(self.chains.keys())
            hotel = np.random.choice(self.chains[chain2].keys())
            while hotel in classes:
                hotel = np.random.choice(self.chains[chain2].keys())
            classes[iy] = hotel
            chains[iy] = chain2

        ims = []
        labels = [c for c in classes for ix in range(self.numPos)]
        for cls,chain in zip(classes,chains):
            clsPaths = np.array(self.chains[chain][cls]['ims'])
            clsSources = np.array(self.chains[chain][cls]['sources'])
            tcamInds = np.where(clsSources=='tcam')[0]
            exInds = np.where(clsSources=='expedia')[0]
            if len(tcamInds) >= self.numPos/2 and len(exInds) >= self.numPos/2:
                numTcam = self.numPos/2
                numEx = self.numPos - numTcam
            elif len(tcamInds) >= self.numPos/2 and len(exInds) < self.numPos/2:
                numEx = len(exInds)
                numTcam = self.numPos - numEx
            else:
                numTcam = len(tcamInds)
                numEx = self.numPos - numTcam

            random.shuffle(tcamInds)
            random.shuffle(exInds)
            ims.extend(list(clsPaths[tcamInds[:numTcam]]))
            ims.extend(list(clsPaths[exInds[:numEx]]))

        batch = self.getProcessedImages(ims)
        chains = [c for c in chains for ix in range(self.numPos)]
        return batch, labels, chains, ims

    def getProcessedImages(self,ims):
        numIms = len(ims)
        imgs = np.array([cv2.resize(cv2.imread(image_file),(self.image_size[0], self.image_size[1])) for image_file in ims])

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            imgs = np.array([cv2.flip(img,1) if random.random() > 0.5 else img for img in imgs])

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(0,self.image_size[0] - self.crop_size[0],numIms)
            left = np.random.randint(0,self.image_size[1] - self.crop_size[1],numIms)
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        imgs = np.array([imgs[ix,top[ix]:(top[ix]+self.crop_size[0]),left[ix]:(left[ix]+self.crop_size[1]),:]-self.meanImage for ix in range(numIms)])
        return imgs
