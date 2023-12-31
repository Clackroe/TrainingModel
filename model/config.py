import os.path
from pathlib import Path

# get the absolute path to the repository so we don't have to deal with relative paths
repo_dir = Path(__file__).parent.parent.absolute()

# Name of the dataset to run CRAFT on. This is not directly used by the model, it is instead used by
corpus_name = "LargeData"
# this config file to define certain input and output locations. You can, of course, override those
# location settings directly and thus completely ignore this setting, it is just useful to use this
# setting to keep everything consistent and organized :)
# Note that in addition to the default setting of "wikiconv" you can also set this to "cmv" and still
# have the code work out-of-the-box (with Reddit CMV data) as the repo includes all support files needed
# for both the Wikiconv and Reddit CMV corpora.
# Name of the conversation metadata field in the ConvoKit corpus to use as the label for training and evaluation.
# Note that the if-statement in the default value is only there to enable users to switch to the CMV data by only
# changing the corpus_name (rather than having to directly change label_metadata as well). If you are using a custom
# corpus, then of course the if-statement is not needed and you can just directly put the name of the metadata
# field in your corpus that you want to use as the label.
label_metadata = "heated" if corpus_name == "LargeData" else "has_removed_comment"

# define file locations
# Where to save the pre-trained model
save_dir = os.path.join(repo_dir, "saved_models", corpus_name)
# File containing unlabeled data for pre-training
train_path = os.path.join(repo_dir, "nn_input_data",
                          corpus_name, "train_processed_dialogs.txt")
# These two files jointly define the
word2index_path = os.path.join(
    repo_dir, "nn_preprocessing", corpus_name, "word2index.json")
index2word_path = os.path.join(
    repo_dir, "nn_preprocessing", corpus_name, "index2word.json")  # model's vocabulary

# Configure model architecture parameters
attn_model = 'general'
MAX_LENGTH = 80  # Maximum sentence length to consider
CONTEXT_SIZE = 16  # Maximum conversational context length to consider
hidden_size = 500  # Hidden size of the utterance and context embeddings
encoder_n_layers = 2  # Number of layers in the utterance encoder
context_encoder_n_layers = 2  # Number of layers in the context encoder
decoder_n_layers = 2  # Number of layers in the decoder
dropout = 0.1  # Dropout rate
batch_size = 2  # Number of conversations per batch

# Configure training/optimization
# Number of pre-training epochs. Smaller by default for wikiconv since it has a larger training corpus
pretrain_epochs = 3 if corpus_name == "LargeData" else 6
finetune_epochs = 30  # Number of fine-tuning epochs
clip = 50.0  # Maximum gradient cutoff during training
# How often to use ground-truth instead of generated output when training the decoder during pre-training phase
teacher_forcing_ratio = 1.0
learning_rate = 0.0001  # Learning rate to use during pre-training
labeled_learning_rate = 1e-5  # Learning rate to use during fine-tuning
decoder_learning_ratio = 5.0  # Learning rate multiplier on the decolder layers
# How often to print output to the screen (measured in training iterations)
print_every = 10
# CRAFT score above which the forecast is considered positive. The default values were tuned on validation data for each corpus.
forecast_thresh = 0.570617 if corpus_name == "wikiconv" else 0.707984

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token
