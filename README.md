## README.md: Question Answering with Seq2Seq LSTM

This notebook demonstrates how to build a basic Sequence-to-Sequence (Seq2Seq) model using LSTMs with a SentencePiece tokenizer for a Question Answering (QA) task.

### Project Overview

The goal of this project is to train a model that can take a natural language question as input and generate a corresponding answer. We use a Seq2Seq architecture, which is well-suited for such tasks, combining an Encoder to process the input question and a Decoder to generate the output answer.

### Setup and Data Loading

1.  **Install Libraries**: Essential libraries like `torchtext`, `nltk`, `pandas`, `sklearn` are installed.
2.  **Load Data**: A JSON dataset (`slmqa.json`) containing 'question' and 'answer' pairs is loaded into a Pandas DataFrame.
3.  **Data Splitting**: The dataset is split into training, validation, and test sets using `train_test_split`.

### Tokenization with SentencePiece

1.  **NLTK for Word Tokenization**: `nltk`'s `word_tokenize` is used to build a vocabulary from the combined questions and answers.
2.  **SentencePiece Model Training**: A SentencePiece model (`qa_spm.model`) is trained on the corpus created from the vocabulary words. This model handles subword tokenization, which is crucial for handling out-of-vocabulary words and improving generalization.
3.  **`TokenizeDataset`**: A custom PyTorch `Dataset` class, `TokenizeDataset`, is defined to prepare data for the model. It tokenizes questions and answers using the trained SentencePiece model and includes special tokens for beginning-of-sequence (`bos_id`), end-of-sequence (`eos_id`), and padding (`pad_id`).
4.  **`collate_fn`**: A `collate_fn` is implemented for the `DataLoader` to handle batching. It pads sequences to the maximum length within each batch and creates `encoder_input`, `decoder_input`, and `target_ids` tensors.
5.  **`DataLoader`**: PyTorch `DataLoader` instances are created for training and validation, utilizing the `TokenizeDataset` and `collate_fn`.

### Model Architecture (Seq2Seq LSTM)

The model consists of three main components:

1.  **`Encoder`**: An LSTM-based encoder that processes the input question. It takes an embedding of the question tokens and outputs the final hidden and cell states, which represent the context of the question.
2.  **`Decoder`**: An LSTM-based decoder that generates the answer. It takes the output of the encoder (hidden and cell states) and the previous target token to predict the next token in the answer sequence. It includes a linear layer (`fc_out`) to map LSTM outputs to vocabulary size.
3.  **`Seq2Seq`**: The main model that combines the Encoder and Decoder. The `forward` method passes the source (question) through the encoder to get context vectors, then uses these to initialize the decoder, which generates the target (answer) sequence.

### Training

1.  **Hyperparameters**: `vocab_size`, `emb_dim`, `hidden_dim`, `n_layers`, and `dropout` are defined.
2.  **Model Instantiation**: `Encoder`, `Decoder`, and `Seq2Seq` models are instantiated.
3.  **Loss Function**: `nn.CrossEntropyLoss` is used, with `ignore_index=-100` to disregard padding tokens in loss calculation.
4.  **Optimizer**: `optim.Adam` is used to optimize the model parameters.
5.  **Training Loop**: The model is trained for a specified number of epochs. For each epoch, it iterates through the `train_loader`, performs forward and backward passes, and updates weights. The average loss per epoch is printed.

### Inference (Generating Answers)

1.  **`generate_answer` function**: This function takes a question string and a `max_len` as input.
2.  **Process Input**: It tokenizes the input question using SentencePiece and passes it through the encoder to obtain initial hidden and cell states.
3.  **Decoding**: The decoder then generates the answer token by token, starting with a `bos_id` and stopping when an `eos_id` is generated or `max_len` is reached.
4.  **Output**: The generated token IDs are then decoded back into a human-readable answer string.

### How to Run

1.  **Execute Cells Sequentially**: Run all code cells in the notebook from top to bottom.
2.  **Input Data**: Ensure `slmqa.json` is available in the Colab environment.
3.  **Train Model**: The training loop will run for `num_epochs` (default 150).
4.  **Generate Answers**: Use the `generate_answer` function with your desired questions.
5.  **Save Model**: The trained model weights are saved to `qa_both_weights.pth`.
