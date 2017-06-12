import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

location = "/media/pc12/DATA/final_dataset/"  # Location of the dataset
sets = [location+"set1/",
        location+"set2/",
        location+"set3/"]  # Locations of the images

items = ["pipe",
         "cable",
         "diver"]
itemNumbers=[0,0,0]

verbose = 0

def findTheFiles():
    total_number_of_images = 0
    nameDataset = {}
    for i in range(3):
        if verbose == 1:
            print("Set {0}:".format(i+1))
        for root, dirs, files in os.walk(sets[i]):
            for file in files:
                if file.endswith(".png"):
                    file_name_with_path = os.path.join(root, file)
                    if verbose == 1:
                        print(file_name_with_path)
                    nameDataset.update({file_name_with_path:[-1,-1,-1]})  # [PipeLabel, CableLabel, DiverLabel]
                    total_number_of_images += 1

        for j, item in enumerate(items):
            ##Cable Labels correction
            positive_list = sets[i]+item+"corrected.txt"
            #print(positive_list)
            image_list = open(positive_list,'r')
            for image_name in image_list:
                image_name_splited = image_name.split("\n")  # get rid of \n
                # print(image_name_splited)
                image_name_itself = sets[i]+image_name_splited[0]
                if nameDataset[image_name_itself] != None:
                    labels = nameDataset[image_name_itself]
                    labels[i] = 1
                    itemNumbers[j] +=1
    return total_number_of_images, nameDataset

def showSomeSamples(dict):
    numberofimages = len(dict.keys())
    items = []
    for i in range(10):
        random_location = int(np.random.rand(1,1) * numberofimages)
        print(random_location)
        print(list(dict.keys())[random_location])
        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(mpimg.imread(list(dict.keys())[random_location]))
        plt.subplot(222)
        plt.imshow(rnd.random((100, 100)))
        plt.subplot(223)
        plt.imshow(rnd.random((100, 100)))
        plt.subplot(224)
        plt.imshow(rnd.random((100, 100)))

        plt.subplot_tool()
        plt.show()


totalNumberOfImages, nameDataset = findTheFiles()
showSomeSamples(nameDataset)
if verbose == 1:
    print("There are total {0} images in the dataset".format(totalNumberOfImages))
    for i, item in enumerate(itemNumbers):
        print("There are {0} {1}-containing image".format(item, items[i]))

    for item in nameDataset.keys():
        print(item + "-->")
        print(nameDataset[item])