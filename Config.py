class Config(object):
    def __init__(self,args,config_preprocess):
        self._embedding_size = args.embedding_size
        self._rnn_size_char = args.rnn_size_char
        self._rnn_size_sent = args.rnn_size_sent
        self._sent_length = config_preprocess['sent_length']
        self._sent_num = config_preprocess['sent_num']
        self._vocab_size = config_preprocess['vocab_size']
        self._num_labels = config_preprocess['num_labels']
        self._batch_size = args.batch_size

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def rnn_size_char(self):
        return self._rnn_size_char

    @property
    def rnn_size_sent(self):
        return self._rnn_size_sent
    @property
    def sent_length(self):
        return self._sent_length

    @property
    def sent_num(self):
        return self._sent_num

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def batch_size(self):
        return self._batch_size
