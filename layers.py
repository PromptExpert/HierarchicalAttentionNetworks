'''
sub-components of the hierachical attention network
'''
import tensorflow as tf

class CharEmbedding(object):
    def __init__(self,vocabulary_size,embedding_size):
        self.embeddings_matrix = tf.get_variable('embeddings_matrix', [vocabulary_size, embedding_size],dtype = tf.float32) #创建一个词嵌入矩阵

    def __call__(self,doc_batch): 
        return tf.nn.embedding_lookup(self.embeddings_matrix,doc_batch) 



class Encoder(object): 
    def __init__(self,rnn_size,maximum,last_size,scope):
        self.rnn_size = rnn_size
        self.maximum = maximum
        self.last_size = last_size
        self.gru_cell_fw = tf.contrib.rnn.GRUCell(self.rnn_size)
        self.gru_cell_bw = tf.contrib.rnn.GRUCell(self.rnn_size)
        self.scope = scope

    def __call__(self,inputs,lengths):
        inputs = tf.reshape(inputs, [-1, self.maximum, self.last_size])
        ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.gru_cell_fw,
                                                                             cell_bw=self.gru_cell_bw,
                                                                             inputs=inputs,
                                                                             dtype=tf.float32,
                                                                             scope = self.scope,
                                                                             sequence_length = lengths)
        annotations = tf.concat((fw_outputs, bw_outputs), 2) 
        return annotations


class Attention(object):
    def __init__(self,rnn_size,maximum,name):
        self.rnn_size = rnn_size
        self.maximum = maximum
        self._u = tf.get_variable(name,[rnn_size*2], dtype = tf.float32)

    def __call__(self,annotations):
        u = tf.contrib.layers.fully_connected(annotations, self.rnn_size*2, activation_fn=tf.nn.tanh) 
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(u,self._u), axis=2)) 
        alpha = tf.expand_dims(alpha, 2) 
        outputs = tf.reduce_sum(tf.multiply(annotations, alpha), axis=1)
        return outputs 

class Projection(object):
    def __init__(self,num_labels):
        self.num_labels = num_labels
        
    def __call__(self,batch_doc_vector):
        logits = tf.contrib.layers.fully_connected(batch_doc_vector, self.num_labels, activation_fn=None)
        return logits 











