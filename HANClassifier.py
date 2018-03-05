'''
Final model,final computing graph of the hierachical attention network
'''
from layers import *#CharEmbedding,CharEncoder,CharAttention,SentEncoder,SentAttention,Projection

def lengths_char(batch):
    '''
    compute the actual lengths for each of the sequences in the batch,used for  char_encoder bidirectional_dynamic_rnn
    the input shape is (batch, max_num_snet, max_length_snet)
    return shape is (batch*sent_num,)
    '''
    batch = tf.sign(batch)
    lengths = tf.reduce_sum(batch,2)
    return tf.reshape(lengths,[-1])

def lengths_sent(batch):
    '''
    compute the actual lengths for each of the sequences in the batch,used for  sent_encoder bidirectional_dynamic_rnn
    the input shape is (batch, max_num_snet, max_length_snet)
    return shape is (batch,)
    '''
    batch = tf.reduce_sum(batch,2)
    batch = tf.sign(batch)
    return tf.reduce_sum(batch,1)




class HANClassifier(object):
    def __init__(self,config):
        self.char_embedding = CharEmbedding(config.vocab_size,config.embedding_size)
        self.char_encoder = Encoder(config.rnn_size_char,config.sent_length,config.embedding_size,'character_encoder')
        self.char_attention = Attention(config.rnn_size_char,config.sent_length,'u_c')
        self.sent_encoder = Encoder(config.rnn_size_sent,config.sent_num,config.rnn_size_char*2,'sentence_encoder')
        self.sent_attention = Attention(config.rnn_size_sent,config.sent_num,'u_s')
        self.project = Projection(config.num_labels)

    def __call__(self,doc_batch):
        lengths_c = lengths_char(doc_batch)
        lengths_s = lengths_sent(doc_batch)
        batch_doc_embeddings = self.char_embedding(doc_batch)
        batch_annotations = self.char_encoder(batch_doc_embeddings,lengths_c)
        batch_sent_vectors = self.char_attention(batch_annotations)
        batch_annotations_sent = self.sent_encoder(batch_sent_vectors,lengths_s)
        batch_doc_vector = self.sent_attention(batch_annotations_sent)
        logits = self.project(batch_doc_vector)

        return logits
