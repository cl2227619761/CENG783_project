import tensorflow as tf
import numpy as np

class WSCNN:
    def __init__(self, weight_file, sess=None, s=4):  # s is rescaling list variable
        self.weightsAndBiases = np.load(weight_file)
        self.keys = sorted(self.weightsAndBiases.keys())
        print(self.keys)
        self.rescaling_factors = np.array([0.5, 0.75, 1, 1.5])
        self.original_image_size = np.array([320, 320], dtype=np.int32)
        self.rescaled_sizes = np.zeros(shape=[s, 2])
        for i in range(s):
            self.rescaled_sizes[i] = np.array(self.original_image_size*self.rescaling_factors[i], dtype=np.uint32)

        self.networkInitializer(sess, verbose=1)
        print(self.rescaled_sizes)
        self.weightsAndBiases = None  # trash out the unnecessary variable to save memory

        self.batch = tf.placeholder(dtype=tf.uint8, shape=[None, 240, 320, 3], name="Input_batch")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="True_labels")
        self.calculateScores()

    def networkInitializer(self, session,verbose=0):
        with tf.device("/gpu:0"):
            self.weights = {
                'conv1_1': tf.Variable(self.weightsAndBiases['conv1_1_W']),
                'conv1_2': tf.Variable(self.weightsAndBiases['conv1_2_W']),
                'conv2_1': tf.Variable(self.weightsAndBiases['conv2_1_W']),
                'conv2_2': tf.Variable(self.weightsAndBiases['conv2_2_W']),
                'conv3_1': tf.Variable(self.weightsAndBiases['conv3_1_W']),
                'conv3_2': tf.Variable(self.weightsAndBiases['conv3_2_W']),
                'conv3_3': tf.Variable(self.weightsAndBiases['conv3_3_W'])

                # 'conv4_1': tf.Variable(self.weightsAndBiases['conv4_1_W']),
                # 'conv4_2': tf.Variable(self.weightsAndBiases['conv4_2_W']),
                # 'conv4_3': tf.Variable(self.weightsAndBiases['conv4_3_W']),
                # 'conv5_1': tf.Variable(self.weightsAndBiases['conv5_1_W']),
                # 'conv5_2': tf.Variable(self.weightsAndBiases['conv5_2_W']),
                # 'conv5_3': tf.Variable(self.weightsAndBiases['conv5_3_W'])
            }
            self.biases = {
                'bias1_1': tf.Variable(self.weightsAndBiases['conv1_1_b']),
                'bias1_2': tf.Variable(self.weightsAndBiases['conv1_2_b']),
                'bias2_1': tf.Variable(self.weightsAndBiases['conv2_1_b']),
                'bias2_2': tf.Variable(self.weightsAndBiases['conv2_2_b']),
                'bias3_1': tf.Variable(self.weightsAndBiases['conv3_1_b']),
                'bias3_2': tf.Variable(self.weightsAndBiases['conv3_2_b']),
                'bias3_3': tf.Variable(self.weightsAndBiases['conv3_3_b'])

                # 'bias4_1': tf.Variable(self.weightsAndBiases['conv4_1_b']),
                # 'bias4_2': tf.Variable(self.weightsAndBiases['conv4_2_b']),
                # 'bias4_3': tf.Variable(self.weightsAndBiases['conv4_3_b']),
                # 'bias5_1': tf.Variable(self.weightsAndBiases['conv5_1_b']),
                # 'bias5_2': tf.Variable(self.weightsAndBiases['conv5_2_b']),
                # 'bias5_3': tf.Variable(self.weightsAndBiases['conv5_3_b'])
            }
            self.conv_out1 = tf.Variable(
                                        tf.constant(0.5, shape=[10, 10, 256, 1024], dtype=tf.float32),
                                        trainable=True, name='Conv_like_fully1')

            self.bias_out1 = tf.Variable(
                                        tf.constant(1e-7, shape=[1024], dtype=tf.float32),
                                        trainable=True, name='Conv_like_fully1_bias')

            self.conv_out2 = tf.Variable(
                                        tf.constant(1e-7, shape=[1, 1, 1024, 3], dtype=tf.float32),
                                        trainable=True, name='Conv_like_fully2')

            self.bias_out2 = tf.Variable(
                                        tf.constant(0.0, shape=[3], dtype=tf.float32),
                                        trainable=True, name='Conv_like_fully2_bias')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)

        if verbose == 1:
            for key in self.weights.keys():
                print("Name: {}, Shape = {}".format(key, self.weights[key].get_shape()))

    def calculateScores(self):

        # PREPROCESSING WILL COME HERE
        # first convert images back to float32, convolution requires that
        images = tf.image.convert_image_dtype(self.batch, dtype=tf.float32, name="uint8_2_float")*(1./255)-0.5
        print(images.get_shape())

        # zero padding from 240*320 to 320*320 for each image
        image_padded = tf.pad(images, [[0, 0], [40, 40], [0, 0], [0, 0]], "CONSTANT", name="320X320_padding")
        print(image_padded.get_shape())

        rescaled_images_0 = tf.image.resize_images(image_padded, self.rescaled_sizes[0])
        rescaled_images_1 = tf.image.resize_images(image_padded, self.rescaled_sizes[1])
        rescaled_images_2 = tf.image.resize_images(image_padded, self.rescaled_sizes[2])
        rescaled_images_3 = tf.image.resize_images(image_padded, self.rescaled_sizes[3])

        netScores = tf.stack([
                            self.network(rescaled_images_0),
                            self.network(rescaled_images_1),
                            self.network(rescaled_images_2),
                            self.network(rescaled_images_3)], axis=1
        )

        self.overAllScore = tf.reduce_mean(netScores, axis=1)
        print(self.overAllScore.get_shape())

    def network(self, processed_batch, verbose=0):
        ################################
        # Convolution 1
        ################################
        conv = tf.nn.conv2d(processed_batch, self.weights['conv1_1'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias1_1'])
        conv1_1 = tf.nn.relu(out, name='Convolutional1_1')

        conv = tf.nn.conv2d(conv1_1, self.weights['conv1_2'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias1_2'])
        conv1_2 = tf.nn.relu(out, name='Convolutional1_1')

        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='MaxPool2by2_1')
        ################################
        # Convolution 2
        ################################
        conv = tf.nn.conv2d(pool1, self.weights['conv2_1'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias2_1'])
        conv2_1 = tf.nn.relu(out, name='Convolutional2_1')

        conv = tf.nn.conv2d(conv2_1, self.weights['conv2_2'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias2_2'])
        conv2_2 = tf.nn.relu(out, name='Convolutional2_2')

        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='MaxPool2by2_2')

        ################################
        # Convolution 3
        ################################
        conv = tf.nn.conv2d(pool2, self.weights['conv3_1'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias3_1'])
        conv3_1 = tf.nn.relu(out, name='Convolutional3_1')

        conv = tf.nn.conv2d(conv3_1, self.weights['conv3_2'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias3_2'])
        conv3_2 = tf.nn.relu(out, name='Convolutional3_2')

        conv = tf.nn.conv2d(conv3_2, self.weights['conv3_3'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases['bias3_3'])
        conv3_3 = tf.nn.relu(out, name='Convolutional3_3')

        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='MaxPool2by2_3')

        conv = tf.nn.conv2d(pool3, self.conv_out1, [1, 1, 1, 1], padding="VALID")
        out = tf.nn.bias_add(conv,self.bias_out1)
        conv_out1 = tf.nn.relu(conv, name="FullyOut1")

        conv = tf.nn.conv2d(conv_out1, self.conv_out2, [1, 1, 1, 1], padding="VALID")
        out = tf.nn.bias_add(conv, self.bias_out2)
        conv_out2 = tf.nn.relu(conv, name="FullyOut2")

        global_max_pool = tf.reduce_max(conv_out2, [1, 2], name="GlobalMaxPool")

        if verbose == 1:
            print(conv1_1.get_shape())
            print(conv1_2.get_shape())
            print(pool1.get_shape())
            print(conv2_1.get_shape())
            print(conv2_2.get_shape())
            print(pool2.get_shape())
            print(conv3_1.get_shape())
            print(conv3_2.get_shape())
            print(conv3_3.get_shape())
            print(conv_out1.get_shape())
            print(conv_out2.get_shape())
            print(global_max_pool.get_shape())

        return global_max_pool

# if __name__ == '__main__':
#     sess = tf.Session()
#     set = tf.placeholder(tf.float32, [None, None, None, 3])
#     WSCNN = WSCNN(set, "./vgg16_weights.npz", sess)
#     print(WSCNN.weights['conv1_1'].get_shape())
#     print(sess.run(WSCNN.weights['conv1_1']))
#     sess.close()
