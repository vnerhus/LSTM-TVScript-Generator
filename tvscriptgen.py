import tensorflow as tf
import numpy as np
import os

from collections import Counter

# -- TEXT IMPORT -- #
def text_import(file_path):
    input_file = os.path.join(file_path)
    with open(input_file, "r", errors='ignore') as f:
        text = f.read()
        text = text[81:]
    return text

text = text_import('./data/simpsons/moes_tavern_lines.txt')

# -- PREPROCESSING -- #
# Lookup table
def create_lookup_tables(text):
    """
    Creates lookup table for vocabulary
    :param text: The text of tv scripts split into words.
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 0)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# Tokenization
def token_lookup():
    """
    Generates a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """

    tlookup = {'.': '||Period||',
        ',': '||Comma||',
        '"': '||Quotation_Mark||',
        ';': '||Semicolon||',
        '!': '||Exclamation_Mark||',
        '?': '||Question_Mark||',
        '(': '||Left_Parentheses||',
        ')': '||Right_Parentheses||',
        '--': '||Dash||',
        '\n': '||Return||'}
    return tlookup

#General preprocessing
def preprocess(text):
    """
    Preprocesses and saves data
    :return: Tuple (token_dict, vocab_to_int, int_to_vocab, int_text)
    """

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]

    return token_dict, vocab_to_int, int_to_vocab, int_text

# -- BUILD MODEL -- #

token_dict, vocab_to_int, int_to_vocab, int_text = preprocess(text)

# Preparing inputs
def get_inputs():
    """
    Creates TF Placeholders for input (x), targets (y), and learning rate (alpha).
    :return: Tuple (input, targets, learning rate)
    """

    x = tf.placeholder(tf.int32, [None, None], name='input')
    y = tf.placeholder(tf.int32, [None, None], name='targets')
    alpha = tf.placeholder(tf.float32, name='learning_rate')
    return x, y, alpha

# Initialize RNN with lstm-cells
def get_init_cell(batch_size, lstm_size, stack_size):
    """
    Creates and initializes a MultiRNNCell.
    :param batch_size: Size of batches.
    :param lstm_size: size of RNN cells (lstm).
    :param stack_size: Number of stacked RNN cells (lstm).
    :return: Tuple (lstmStack, initial_state)
    """

    #Single Cell
    #lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    #Cellstack:
    lstmStack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(stack_size)])
    #initialization and naming
    initial_state = lstmStack.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, 'initial_state')
    return lstmStack, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Creates embedding for input data.
    :param input_data: TF Placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions.
    :return: Embedded input.
    """

    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def build_rnn(cell, embed):
    """
    Creates an RNN using a RNN cell.
    :param cell: RNN Cell.
    :param inputs: Input text data.
    :return: Tuple (outputs, final_state)
    """

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    #Name state
    final_state = tf.identity(final_state, 'final_state')

    return outputs, final_state

def build_nn(cell, lstm_size, input_data, vocab_size, embed_dim):
    """
    Builds part of the neural network.
    :param cell: RNN Cell.
    :param lstm_size: Size of RNN Cell (lstm).
    :param input_data: Input data.
    :param vocab_size: Vocabulary size.
    :param embed_dim: Number of embedding dimensions.
    :return: Tuple (logits, final_state)
    """

    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)

    #Fully connected layer, with ReLU activation.
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))

    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    """
    Divides input and target into batches.
    :param int_text: Text with the words replaced by their ids.
    :param batch_size: The size of batch.
    :param seq_length: the length of a sequence.
    :return: Batches as Numpy array
    """

    n_batches = len(int_text)//(batch_size*seq_length)

    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))

# -- TRAINING -- #

# - Hyperparameters - #
# Number of Epochs
num_epochs = 500
# Batch Size
batch_size = 128
# lstm Size
lstm_size = 128
# Stack Size
stack_size = 3
# Embedding Dimension Size
embed_dim = lstm_size
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 11

save_dir = './save'

# - TF Graph - #
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], lstm_size, stack_size)
    logits, final_state = build_nn(cell, lstm_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# - Training sess - #
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if(epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

# -- GENERATING NEW SCRIPT -- #
def get_tensors(loaded_graph):
    """
    Gets input, initial_state, final_state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file.
    :return Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """

    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor= loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

def pick_word(probabilities, int_to_vocab):
    """
    Picks the next word in the generated text.
    :param probabilities: Probabilities of the next word.
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values.
    :return: String of the predicted word.
    """

    return np.random.choice(list(int_to_vocab.values()), p=probabilities)

gen_length = 200
# homer_simpson, moe_szyslak, or barney_gumble
#prime_word = 'moe_szyslak'
prime_word = 'thomas'

#loaded_graph = tf.Graph()
loaded_graph = train_graph
with tf.Session(graph=train_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + '']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run([probs, final_state], {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)
