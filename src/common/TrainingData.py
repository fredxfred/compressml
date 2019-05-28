import pathlib
import tensorflow as tf
import random
#https://www.tensorflow.org/tutorials/load_data/images

class TrainingData:
    
    default_height = 480
    default_width = 360
    
    def __init__(self, path, pathglob="*", height=default_height, width=default_width, \
                 resize_channels=True, grayscale=False):
        self.filenames = tf.placeholder(tf.string, shape=[None])
        self.dataset = tf.data.TFRecordDataset(self.filenames)
        self.paths = [p.as_posix() for p in list(pathlib.Path(path).glob(pathglob))]
        random.shuffle(self.paths)
        self.height = height
        self.width = width
        self.resize_channels = resize_channels
        self.grayscale = grayscale
        
    def default_parse(self, filename):
        img = tf.image.decode_image(tf.read_file(filename))
        img = tf.cast(img, tf.float32)
        if path.endswith('.gif'): #it's a 4d tensor
            img = img[0,...]
        if self.resize_channels:
            img /= 255.0
        img = image.resize_image_with_crop_or_pad(img, self.height, self.width)
        if self.grayscale:
            img = image.rgb_to_grayscale(img)
            img = img[..., 0]
        return img, img
        
    def start_sess(self, sess, batchsize = 32, parser=None):
        if parser == None:
            parser = self.default_parse
        self.dataset = self.dataset.map(parser)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(batchsize)
        return sess.run(self.dataset.make_initializable_iterator().initializer, \
                        feed_dict={self.filenames : self.paths})
