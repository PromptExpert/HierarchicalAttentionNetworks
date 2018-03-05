import tensorflow as tf
import argparse
import os
import pickle
from Config import Config
from HANClassifier import HANClassifier
import time
import sys
import arg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #解决Warning: The TensorFlow library wasn't compiled to use SSE4.2 instructions
parser = argparse.ArgumentParser()
arg.add_args(parser)
args = parser.parse_args()


def train():
    if args.checkpoint:
        saver.restore(sess,args.checkpoint)
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(args.epochs):
        sess.run(iterator.initializer)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                try:
                    _, loss_ = sess.run([optimizer, loss])
                except tf.errors.InvalidArgumentError:
                    break
                total_loss += loss_
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1:.2f}'.format(i, total_loss/n_batches))
        saver.save(sess, args.models_dir+"{0}th_epoch_model_{1:.2f}.ckpt".format(i,total_loss/n_batches))

def test():
    #compute accuracy of valid data
    saver.restore(sess,args.checkpoint)
    total_correct_preds = 0
    sess.run(iterator.initializer)
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
    print('Accuracy {0:.2f}%'.format(total_correct_preds*100/data_size))

def predict():
    saver.restore(sess,args.checkpoint)
    sess.run(iterator.initializer)
    f = open('predicted.csv','w')
    try:
        while True:
            _preds, _ids = sess.run([pred,ids])
            for i in range(len(_preds)):
               f.write(str(_ids[i],'utf-8')+','+ str(_preds[i])+'\n')
    except tf.errors.OutOfRangeError:
        pass
    f.close()

if __name__ == "__main__":
    ########### Load Config ###########
    config_preprocess = pickle.load(open('preprocess/config_preprocess.pickle', 'rb'))
    config  = Config(args,config_preprocess)

    ########### Load Data ###########
    if args.tiny:
        filename = 'preprocess/tiny_preprocessed.pickle'
    elif args.test:
        filename = 'preprocess/test_preprocessed.pickle'
    elif args.predict:
        filename = 'preprocess/predict_preprocessed.pickle'
    else:
        filename = 'preprocess/train_preprocessed.pickle'

    if args.predict:
        data = pickle.load(open(filename,'rb'))
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.batch(args.batch_size)
        iterator = data.make_initializable_iterator()
        sents,ids = iterator.get_next()
    else:
        data = pickle.load(open(filename,'rb'))
        data_size = len(data[1]) #used in test
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(10000)
        data = data.batch(args.batch_size)
        iterator = data.make_initializable_iterator()
        sents,labels = iterator.get_next()

    ########### Define the inference model ###########
    han_classifier = HANClassifier(config)
    logits = han_classifier(sents)

    ########### Define loss function ###########
    if not args.predict:
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        loss = tf.reduce_mean(entropy)

    ########### Define Optimizer ###########
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss)

    ############ Test Accuracy ###########
    preds = tf.nn.softmax(logits)
    pred  = tf.argmax(preds, 1,output_type = tf.int32) + 1
    if not args.predict:
        correct_preds = tf.equal(tf.argmax(preds, 1,output_type = tf.int32),labels)
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    ############ Create Saver ###########
    saver = tf.train.Saver()

    ########### Training ###########
    with tf.Session() as sess:
        if args.predict:
            predict()
        elif args.test:
            test()
        else:
            train()
