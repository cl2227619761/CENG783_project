import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from skimage import io
location = "/media/pc12/DATA/final_dataset/"  # Location of the dataset
sets = [location+"set1/",
        location+"set2/",
        location+"set3/"]  # Locations of the images

itemNumbers=[0,0,0]

verbose = 1

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
                    nameDataset.update({file_name_with_path:np.array([-1,-1,-1], dtype=np.float32)})  # [PipeLabel, CableLabel, DiverLabel]
                    total_number_of_images += 1


        pipe_file = open(sets[i]+"pipecorrected.txt",'r')
        for line in pipe_file:
            image_name_splited = line.split("\n")  # get rid of \n
            image_name_itself = sets[i] + image_name_splited[0]
            nameDataset[image_name_itself][0] = 1
            itemNumbers[0] +=1
        pipe_file.close()

        cable_file = open(sets[i] + "cablecorrected.txt", 'r')
        for line in cable_file:
            image_name_splited = line.split("\n")  # get rid of \n
            image_name_itself = sets[i] + image_name_splited[0]
            nameDataset[image_name_itself][1] = 1
            itemNumbers[1] += 1
        cable_file.close()

        diver_file = open(sets[i] + "divercorrected.txt", 'r')
        for line in diver_file:
            image_name_splited = line.split("\n")  # get rid of \n
            image_name_itself = sets[i] + image_name_splited[0]
            nameDataset[image_name_itself][2] = 1
            itemNumbers[2] += 1
        diver_file.close()
    return total_number_of_images, nameDataset

def showSomeSamples(dict):
    numberofimages = len(dict.keys())
    items = []
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        random_location = int(np.random.rand(1,1) * numberofimages)
        print(random_location)
        print(list(dict.keys())[random_location])
        print(list(dict.values())[random_location])
        ax.imshow(mpimg.imread(list(dict.keys())[random_location]))
        labelstr=""
        if list(dict.values())[random_location][0] == 1:
            labelstr = labelstr+"Pipe "

        if list(dict.values())[random_location][1] == 1:
            labelstr = labelstr+"Cable "

        if list(dict.values())[random_location][2] == 1:
            labelstr = labelstr+"Diver "

        ax.set_xlabel(labelstr)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def showObtainedImages(imageArray,labelArray):
    numberofimages = len(imageArray)

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        random_location = int(np.random.rand(1, 1) * numberofimages)
        ax.imshow(imageArray[random_location])
        labelstr=""
        if labelArray[random_location,0] == 1:
            labelstr = labelstr+"Pipe "

        if labelArray[random_location,1] == 1:
            labelstr = labelstr+"Cable "

        if labelArray[random_location,2] == 1:
            labelstr = labelstr+"Diver "

        ax.set_xlabel(labelstr)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def writeTfrecords(name_dataset, output_location, how_many_files=1000,totalFile=1,image_count=None):
    iteration=0
    writer = None
    for current_file, key in enumerate(name_dataset.keys()):
        file_name = key
        label = name_dataset[key]
        if verbose == 1:
            print("file: ", file_name)
            print("label: ", label)

        if current_file % how_many_files == 0: # 100 by default

            if writer: #close previously opened writer item
                writer.close()
            record_filename = "{a}maris{b}.tfrecords".format(a=output_location, b=iteration)
            print(record_filename)

            fw = open(record_filename, "w") #to create the file
            fw.close()

            writer = tf.python_io.TFRecordWriter(record_filename)
            print("Generated", record_filename)

            iteration += 1

        if (image_count != None and current_file == totalNumberOfImages) or iteration == totalFile+1:
            break

        #Read the image and decode
        # image_file = tf.read_file(file_name)
        # image = tf.image.decode_png(image_file, channels=3)
        image = io.imread(file_name)
        image_resized = image[0:240,0:320]  # crop unnecessary pixels
        # fig = plt.figure()
        # plt.imshow(image_resized)
        # plt.show()
        # image_flat = np.reshape(image_resized,newshape=[240*320*3])
        image_bytes = image_resized.tobytes()
        label_bytes = label.tobytes()
        print(label.shape, image_resized.shape)

        feature = tf.train.Features(
            feature = {'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),

                       'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}
        )
        example = tf.train.Example(features = feature)
        writer.write(example.SerializeToString())

    writer.close()

def readAndDecode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                                                            'label': tf.FixedLenFeature([], tf.string),
                                                            'image': tf.FixedLenFeature([], tf.string)
                                                            })
    image_flat = tf.decode_raw(features['image'], np.uint8)
    label_flat = tf.decode_raw(features['label'], np.float32)
    image = tf.reshape(image_flat,[240,320,3])
    label = tf.reshape(label_flat,[3,])

    return image, label


def getBatchFromFile(FILE,n_batch):
    sess = tf.InteractiveSession()
    image_batch = np.zeros(shape=[n_batch, 240, 320, 3])
    label_batch = np.zeros(shape=[n_batch, 3])
    filename_queue = tf.train.string_input_producer(["/media/pc12/DATA/final_dataset/tf/maris0.tfrecords"])
    with tf.device('/cpu:0'):
        image, label= readAndDecode(filename_queue)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(0, n_batch):
            example, l= sess.run([image, label])
            image_batch[i] = example
            label_batch[i] = l
        coord.request_stop()
        coord.join(threads)
    sess.close()
    return image_batch,label_batch

totalNumberOfImages, nameDataset = findTheFiles()
# showSomeSamples(nameDataset)
if verbose == 1:
    items = ["pipe","cable","diver"]
    print("There are total {0} images in the dataset".format(totalNumberOfImages))
    for i, item in enumerate(itemNumbers):
        print("There are {0} {1}-containing image".format(item, items[i]))

    # for item in nameDataset.keys():
    #     print(item + "-->")
    #     print(nameDataset[item])

# writeTfrecords(nameDataset, "/media/pc12/DATA/final_dataset/tf/", image_count=totalNumberOfImages)
imageBatch,labelBatch = getBatchFromFile("/media/pc12/DATA/final_dataset/tf/",1000)
print(imageBatch[800,:,:,1])
image11 = io.imread("/media/pc12/DATA/final_dataset/set1/frame01641_set1_0.png")
print(image11[:,:,2])
showObtainedImages(imageBatch,  labelBatch)