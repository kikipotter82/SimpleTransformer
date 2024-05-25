import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from typing import Optional
import os
import matplotlib.pyplot as plt
import time
import nltk

nltk.download('punkt')
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, Decoder, Decoder2
from utilities import Utilities
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
dropout = 0.2

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _, _ = classifier(X)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def check_sanity(tokenizer, model, sentence, type):
    utils = Utilities(tokenizer, model)
    utils.sanity_check(sentence, block_size, type)

def count_parameters(model):
    cnt = sum(p.numel() for p in model.parameters())
    return f"Number of Parameters: {cnt}"

def read_data(inputfile,tokenizer):
    with open(inputfile, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = LanguageModelingDataset(tokenizer, text,  block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Speech Recognition Model Training and Testing')
    parser.add_argument('--part', type=str, choices=['part1', 'part2','part3'], required=True, help='Select which part of the model to run: encoder for part 1, decoder for part 2')
    args = parser.parse_args()

    if args.part == 'part1':
        run_encoder_part()
    elif args.part == 'part2':
        run_decoder_part()
    elif args.part == 'part3':
        part3()

def run_encoder_part():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # Part 1: Encoder Trained With Classifier
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    
    encoder = Encoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, dropout, 
                      n_input, n_hidden, n_output).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
    acc_train = []
    acc_test = []
    loss_list = [] 
    
    # for the classification task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        total_loss = 0
        num_batches = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            logits, loss, _ = encoder(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        loss_list.append(average_loss)

        train_acc = compute_classifier_accuracy(encoder, train_CLS_loader)
        test_acc = compute_classifier_accuracy(encoder, test_CLS_loader)

        acc_train.append(train_acc)
        acc_test.append(test_acc)

        print(f"Epoch {epoch + 1}/{epochs_CLS}, Train Loss: {average_loss:.4f}, Train Acc: {train_acc}%, Test Acc: {test_acc}%")

    print("Train Accuracies: ", acc_train)
    print("Test Accuracies: ", acc_test)

    # plot
        
    # Loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs_CLS + 1), loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs of the Encoder')
    plt.legend()
    plt.grid(True)  
    ax = plt.gca() 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    plt.savefig('encoder_loss.png')  
    plt.show()

    # Training and Testing Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs_CLS + 1), acc_train, label='Training Accuracy', marker='o')  
    plt.plot(range(1, epochs_CLS + 1), acc_test, label='Testing Accuracy', marker='o') 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')  
    plt.title('Training and Testing Accuracy of the Encoder')
    plt.legend()
    plt.grid(True)  
    ax = plt.gca() 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    plt.savefig('encoder_accuracy.png')  
    plt.show()  

    sentence_idx1_1 = "In that box is the bill."
    sentence_idx2_2 = "Clearly, each and every one of us can find fault with something in this agreement."
    check_sanity(tokenizer, encoder, sentence_idx1_1,'Encoder of sentence 1')
    check_sanity(tokenizer, encoder, sentence_idx2_2,'Encoder of sentence 2')
    print(count_parameters(encoder))
    
def run_decoder_part():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # Part 2: Pretraining Decoder Language Model
    inputfile = "speechesdataset/train_LM.txt"
    train_LM_loader = read_data(inputfile, tokenizer)
    # 0: Barack Obama, 1: George W. Bush, 2: George H. Bush.
    test0file = "speechesdataset/test_LM_obama.txt"
    test_0_loader = read_data(test0file, tokenizer)
    test1file = "speechesdataset/test_LM_wbush.txt"
    test_1_loader = read_data(test1file, tokenizer)
    test2file = "speechesdataset/test_LM_hbush.txt"
    test_2_loader = read_data(test2file, tokenizer)

    decoder = Decoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    iters = []
    ppl_tr = []
    ppl_te0 = []
    ppl_te1 = []
    ppl_te2 = []

    # Start timing
    start_time = time.time()
    
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i % eval_interval == 0 and i != 0:
            print(i)
            iters.append(i)
            ppl_tr.append(compute_perplexity(decoder, train_LM_loader))
            ppl_te0.append(compute_perplexity(decoder, test_0_loader))
            ppl_te1.append(compute_perplexity(decoder, test_1_loader))
            ppl_te2.append(compute_perplexity(decoder, test_2_loader))      

        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        logits, loss, _ = decoder(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print("Training Time (seconds): ", training_time)
    print("Train Perplexity: ", ppl_tr)
    print("Test 0: Barack Obama Perplexity: ", ppl_te0)
    print("Test 1: George W. Bush Perplexity: ", ppl_te1)
    print("Test 2: George H. Bush Perplexity: ", ppl_te2)

    # plot
    plt.figure(figsize=(8,6))
    plt.plot(iters, ppl_tr, label='Train Perplexity', color='blue', linestyle='-', marker='o') 
    plt.plot(iters, ppl_te0, label='Test 0: Barack Obama Perplexity', color='green', linestyle='--', marker='o')
    plt.plot(iters, ppl_te1, label='Test 1: George W. Bush Perplexity', color='red', linestyle='-.', marker='o')
    plt.plot(iters, ppl_te2, label='Test 2: George H. Bush Perplexity', color='purple', linestyle=':', marker='o')

    plt.xlabel('Iterations', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title('Decoder Perplexity Across Iterations')  
    plt.grid(True)
    plt.legend()  
    ax = plt.gca() 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    plt.savefig('decoder_perplexity.png')
    plt.show()  

    sentence_1 = "And that's why I leave this stage tonight even more optimistic about this country than when we started."
    sentence_2 = "It determines who our Congress is."
    check_sanity(tokenizer, decoder, sentence_1,'Decoder of sentence 1')
    check_sanity(tokenizer, decoder, sentence_2,'Decoder of sentence 2')
    print(count_parameters(decoder))

def part3():
    lr_new = 0.01
    # lr_new = 0.1
    # lr_new = 1e-3 
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # Part 2: Pretraining Decoder Language Model
    inputfile = "speechesdataset/train_LM.txt"
    train_LM_loader = read_data(inputfile, tokenizer)
    # 0: Barack Obama, 1: George W. Bush, 2: George H. Bush.
    test0file = "speechesdataset/test_LM_obama.txt"
    test_0_loader = read_data(test0file, tokenizer)
    test1file = "speechesdataset/test_LM_wbush.txt"
    test_1_loader = read_data(test1file, tokenizer)
    test2file = "speechesdataset/test_LM_hbush.txt"
    test_2_loader = read_data(test2file, tokenizer)

    decoder = Decoder2(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, dropout, window_size=5).to(device)
    optimizer = torch.optim.Adagrad(decoder.parameters(), lr=lr_new)
    #optimizer = torch.optim.RMSprop(decoder.parameters(), lr=lr_new)
    # optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr_new)
    iters = []
    ppl_tr = []
    ppl_te0 = []
    ppl_te1 = []
    ppl_te2 = []

    # Start timing
    start_time = time.time()

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i % eval_interval == 0 and i != 0:
            print(i)
            iters.append(i)
            ppl_tr.append(compute_perplexity(decoder, train_LM_loader))
            ppl_te0.append(compute_perplexity(decoder, test_0_loader))
            ppl_te1.append(compute_perplexity(decoder, test_1_loader))
            ppl_te2.append(compute_perplexity(decoder, test_2_loader))      

        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        logits, loss, _ = decoder(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print("Training Time (seconds): ", training_time)

    print("Train Perplexity: ", ppl_tr)
    print("Test 0: Barack Obama Perplexity: ", ppl_te0)
    print("Test 1: George W. Bush Perplexity: ", ppl_te1)
    print("Test 2: George H. Bush Perplexity: ", ppl_te2)

    # plot
    plt.figure(figsize=(8,6))
    plt.plot(iters, ppl_tr, label='Train Perplexity', color='blue', linestyle='-', marker='o') 
    plt.plot(iters, ppl_te0, label='Test 0: Barack Obama Perplexity', color='green', linestyle='--', marker='o')
    plt.plot(iters, ppl_te1, label='Test 1: George W. Bush Perplexity', color='red', linestyle='-.', marker='o')
    plt.plot(iters, ppl_te2, label='Test 2: George H. Bush Perplexity', color='purple', linestyle=':', marker='o')

    plt.xlabel('Iterations', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title('Decoder Perplexity Across Iterations')  
    plt.grid(True)
    plt.legend()  
    ax = plt.gca() 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    plt.savefig('decoder_perplexity_part3.png')
    plt.show()  

    sentence_1 = "And that's why I leave this stage tonight even more optimistic about this country than when we started."
    sentence_2 = "It determines who our Congress is."
    check_sanity(tokenizer, decoder, sentence_1,'Decoder_part3 (s1)')
    check_sanity(tokenizer, decoder, sentence_2,'Decoder_part3 (s2)')
    print(count_parameters(decoder))

    
if __name__ == "__main__":
    main()
