import torch
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import math
import os
import sys
import torch.optim as optim
import torch.nn as nn
from queue import PriorityQueue
import matplotlib.pyplot as plt

def beam_search(model, image_tensor, vocab_obj, device, k=3, max_length=200):
    """
    Performs beam search on a neural machine translation model to generate a list of candidate translations.

    Args:
        model: a PyTorch model for neural machine translation.
        image_tensor: a PyTorch tensor representing an image.
        vocab_obj: a Vocabulary object containing the vocabulary used during training.
        device: a string representing the device to run the model on (e.g. 'cuda' or 'cpu').
        k: an integer representing the number of top predictions to return.
        max_length: an integer representing the maximum length of a token.

    Returns:
        A list of k tuples, each containing a sequence of integers representing a translation and its log probability.
    """
    # Initialize the priority queue
    pq = PriorityQueue()

    # Initialize the first k beams
    initial_beam = ([vocab_obj.sos_idx], 0) # (Sequence, Log Probability)
    pq.put(initial_beam)
    completed_beams = []

    for _ in range(max_length):
        new_candidates = []

        # Process the top k candidates
        for _ in range(pq.qsize()):
            beam = pq.get()

            if beam[0][-1] == vocab_obj.eos_idx:
                completed_beams.append(beam)
                continue

            # Prepare the input tensor
            trg_tensor = torch.LongTensor(beam[0]).unsqueeze(1).to(device)
            trg_tensor = trg_tensor.transpose(0, 1)  # Swap the first and second dimensions

            # Get the next token probabilities
            with torch.no_grad():
                output = model(image_tensor, trg_tensor) # (N, seq_len, trg_vocab_size)

            # Calculate log probabilities
            log_probs = torch.log_softmax(output[:, -1], dim=1).squeeze(0)

            # Get the top k candidates
            topk_log_probs, topk_indices = log_probs.topk(k, dim=0)

            for i in range(k):
                new_seq = beam[0] + [topk_indices[i].item()]
                new_log_prob = beam[1] + topk_log_probs[i].item()

                new_candidate = (new_seq, new_log_prob)
                new_candidates.append(new_candidate)

        # Keep only the top k candidates
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        topk_candidates = new_candidates[:k]

        # Add the top k candidates back to the priority queue
        for candidate in topk_candidates:
            pq.put(candidate)

    # Get the k most probable completed beams
    completed_beams.sort(key=lambda x: x[1], reverse=True)
    k_best_beams = completed_beams[:k]

    return k_best_beams

def translate_sentence(model, image, vocab_obj, transform, device, k=3, max_length=200):
    """
    Translates an image caption using a trained neural machine translation model and beam search.

    Args:
        model: a PyTorch model for neural machine translation.
        image: a PIL image or a PyTorch tensor representing an image.
        vocab_obj: a Vocabulary object containing the vocabulary used during training.
        transform: a PyTorch transform to apply to the image (e.g. normalization).
        device: a string representing the device to run the model on (e.g. 'cuda' or 'cpu').
        k: an integer representing the number of top predictions to return.
        max_length: an integer representing the maximum length of a token.

    Returns:
        A list of k lists of strings, each containing the translated caption with the highest probability for the given image.
    """
    # Check if the image is a tensor or a PIL image
    if not torch.is_tensor(image):
        # Convert to Tensor
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        image_tensor = image.to(device)
        image_tensor = image_tensor.unsqueeze(0).to(device)

    # Perform beam search
    k_best_beams = beam_search(model, image_tensor, vocab_obj, device, k, max_length)

    # Convert token sequences to sentences
    translated_sentences = []
    for beam in k_best_beams:
        translated_sentence = [vocab_obj.id2token[token] for token in beam[0]]
        # remove start token
        translated_sentences.append(translated_sentence[1:-1])

    return translated_sentences


# def translate_sentence(model, image, vocab_obj, transform, device, k=3, max_length=200):

#     # Check if the image is a tensor or a PIL image
#     if not torch.is_tensor(image):
#         # Convert to Tensor
#         image_tensor = transform(image)
#         image_tensor = image_tensor.unsqueeze(0).to(device)
#     else:
#         image_tensor = image.to(device)
#         image_tensor = image_tensor.unsqueeze(0).to(device)


#     # print('image_tensor shape: ', image_tensor.shape)

#     outputs = [vocab_obj.sos_idx]
#     for i in range(max_length):
#         trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
#         trg_tensor = trg_tensor.transpose(0, 1)  # Swap the first and second dimensions

#         # print('trg_tensor shape: ', trg_tensor.shape)

#         with torch.no_grad():
#             output = model(image_tensor, trg_tensor) # (N, seq_len, trg_vocab_size)

#         best_guess = output.argmax(2)[:, -1].item()
#         if best_guess == vocab_obj.eos_idx:
#             break

#         outputs.append(best_guess)


#     translated_sentence = [vocab_obj.id2token[token] for token in outputs]
#     # remove start token
#     return translated_sentence[1:]


def check_convergence(losses, threshold=0.01):
    """
    Checks if a model has converged by comparing the difference between the losses of the last two epochs.

    Args:
        losses: a list of floats representing the losses of the model for each epoch.
        threshold: a float representing the threshold of the difference in losses for convergence.

    Returns:
        True if the model has converged, False otherwise.
    """
    if len(losses) < 2:
        return False
    return abs(losses[-1] - losses[-2]) < threshold

def plot_losses_bleu(epoch_losses, validation_bleu):
    """
    Plots the epoch_losses and validation BLEU scores for a model.

    Args:
        epoch_losses: a list of floats representing the epoch losses for the model.
        validation_bleu: a list of floats representing the validation BLEU scores for the model.

    Returns:
        None
    """
    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot the epoch_losses on the left y-axis
    ax1.plot([i+1 for i in range(len(epoch_losses))], epoch_losses, color='tab:red', label='Epoch Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Losses')

    # Instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the validation_bleu scores on the right y-axis
    ax2.plot([i+1 for i in range(len(validation_bleu))], validation_bleu, color='tab:blue', label='Validation BLEU Scores')
    ax2.set_ylabel('Validation BLEU Scores')
    ax2.set_ylim([0, 1.05])

    # Set the x-axis ticks to start from 1 and go up to the length of the data
    ax1.set_xticks(range(1, len(epoch_losses) + 1))

    # Add a title to the plot
    plt.title('Epoch Losses and Validation BLEU Scores vs. Epochs')

    # Create a legend for the plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left')

    # Show the plot
    plt.show()

def train(model, model_name, train_loader, validation_loader, vocab_obj, max_token_length, num_epochs, learning_rate, device, print_every=10):
    """
    Trains a neural machine translation model.

    Args:
        model: a PyTorch model to be trained.
        model_name: a string representing the name of the model.
        train_loader: a PyTorch DataLoader object for the training set.
        validation_loader: a PyTorch DataLoader object for the validation set.
        vocab_obj: a Vocabulary object that contains the vocabulary of the training set.
        max_token_length: an integer representing the maximum length of a token.
        num_epochs: an integer representing the number of epochs to train the model for.
        learning_rate: a float representing the learning rate of the optimizer.
        device: a string representing the device to train the model on (e.g. 'cuda' or 'cpu').
        print_every: an integer representing the number of batches to print training stats for.

    Returns:
        epoch_losses_train: a list of floats representing the losses of the model for each epoch.
        bleu_scores: a list of floats representing the BLEU scores of the model for each epoch.
    """
    # Calculate total batches
    total_step = math.ceil(len(train_loader.dataset) / train_loader.batch_size)

    # Create a log file for the model
    log_file = f"./models/{model_name}/{model_name}_log.txt"

    # Send the model to the specified device
    model.to(device)

    # Create a directory to save the model
    os.makedirs(f"./models/{model_name}", exist_ok=True)

    # Write hyperparameters to log file
    write_hyperparameters_to_log(log_file, model, model_name, learning_rate, train_loader.batch_size, num_epochs, device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pad_idx = vocab_obj.pad_idx
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epoch_losses_train = []
    bleu_scores = []

    for epoch in range(1, num_epochs+1):
        # Set the model to train mode
        model.train()

        # Initialize loss and BLEU score lists
        losses = []
        epoch_loss = 0

        # Print the current epoch number
        print(f"[Epoch {epoch} / {num_epochs}]")

        for batch_idx, (images, label, img_names) in enumerate(train_loader):
            # Convert the target labels to token indices
            token_idxs = [vocab_obj.split_string_by_tokens(label, max_token_length) for label in label]
            target = torch.tensor(token_idxs).to(device) # (N, max_length)

            # Send the input images and target labels to the specified device
            images = images.to(device) # (N, 1, height, width)

            # Forward pass
            output = model(images, target[:, :-1]) # (N, max_length - 1, trg_vocab_size)
            output = output.reshape(-1, output.shape[2]) # (N * max_length - 1, trg_vocab_size)
            target = target[:, 1:].reshape(-1) # (N * max_length - 1)

            # Zero out the gradients
            optimizer.zero_grad()

            # Calculate the loss and append to losses list
            loss = criterion(output, target)
            losses.append(loss.item())
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            # Clip gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Update the model parameters
            optimizer.step()

            # Print training stats
            if batch_idx % print_every == 0:
                stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, batch_idx, total_step, loss.item(), np.exp(loss.item()))
                print('\r' + stats, end="\n")
                sys.stdout.flush()

                # Write training stats to the log file
                with open(log_file, 'a', encoding='windows-1252') as f:
                    f.write(stats + '\n')
                    f.flush()

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join('./models', model_name, '{}-{}.pkl'.format(model_name, epoch)))

        # Calculate and print the epoch loss
        epoch_losses_train.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}")
        # Write the epoch loss to the log file
        with open(log_file, 'a', encoding='windows-1252') as f:
            f.write(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}\n")

        # Evaluate the model on the validation set
        print("Evaluating the model on the validation set...")
        bleu_score = evaluate(model, validation_loader, vocab_obj, device, max_length=max_token_length, print_every=print_every)
        bleu_scores.append(bleu_score)

        # Print and write the BLEU score to the log file
        print("BLEU Score: {}".format(bleu_score))
        with open(log_file, 'a', encoding='windows-1252') as f:
            f.write("Epoch {}, BLEU Score: {}\n".format(epoch, bleu_score))

        # Check for convergence
        if check_convergence(bleu_scores, threshold=0.01):
            print("Converged!")
            break

    # Return the epoch losses and BLEU scores
    return epoch_losses_train, bleu_scores

def write_hyperparameters_to_log(log_file, model, model_name, learning_rate, batch_size, num_epochs, device):
    """
    Writes the hyperparameters of a model to a log file.

    Args:
        log_file: a string representing the path of the log file.
        model: a PyTorch model whose hyperparameters will be logged.
        model_name: a string representing the name of the model.
        learning_rate: a float representing the learning rate of the optimizer.
        batch_size: an integer representing the batch size of the data loader.
        num_epochs: an integer representing the number of epochs to train the model for.
        device: a string representing the device to train the model on (e.g. 'cuda' or 'cpu').

    Returns:
        None
    """
    with open(log_file, 'w', encoding='windows-1252') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Embedding Size: {model.embed_size}\n")
        f.write(f"Resnet Type: {model.resnet_type}\n")
        f.write(f"Number of Layers: {model.num_layers}\n")
        f.write(f"Forward Expansion: {model.forward_expansion}\n")
        f.write(f"Number of Heads: {model.heads}\n")
        f.write(f"Dropout: {model.dropout}\n")
        f.write(f"Max Length: {model.max_length}\n")
        f.write("\n")
        f.flush()

def evaluate(model, data_loader, vocab_obj, device, max_length=200, print_every=10):
    """
    Evaluates a neural machine translation model on a validation or test set.

    Args:
        model: a PyTorch model to be evaluated.
        data_loader: a PyTorch DataLoader object for the validation or test set.
        vocab_obj: a Vocabulary object that contains the vocabulary of the training set.
        device: a string representing the device to evaluate the model on (e.g. 'cuda' or 'cpu').
        max_length: an integer representing the maximum length of a token.
        print_every: an integer representing the number of batches to print BLEU scores for.

    Returns:
        The average BLEU score of the model on the validation or test set.
    """
    # Calculate total batches
    total_step = math.ceil(len(data_loader.dataset) / data_loader.batch_size)

    # Set the model to evaluation mode
    model.eval()

    # Initialize BLEU score to 0
    total_bleu = 0

    with torch.no_grad():
        for batch_idx, (images, label, img_names) in enumerate(data_loader):
            # Convert target labels to token indices
            token_idxs = [vocab_obj.split_string_by_tokens(label, max_length) for label in label]

            # Convert token indices to tensors
            target = torch.tensor(token_idxs).to(device) # (N, max_length)
            images = images.to(device) # (N, 1, height, width)

            # Forward pass
            output = model(images, target) # (N, max_length, trg_vocab_size)

            # Flatten output to (N, max_length), by selecting max probability index
            output = output.argmax(2) # (N, max_length)

            # Convert tensors to a list of tokens
            pred_token_idxs = output.tolist() # (N, max_length)

            # Convert token indices to strings
            batch_ref = [vocab_obj.id2string(token_idx) for token_idx in token_idxs]
            batch_hyp = [vocab_obj.id2string(pred_token_idx) for pred_token_idx in pred_token_idxs]

            # Calculate BLEU score for the batch
            batch_bleu = 0
            for ref, hyp in zip(batch_ref, batch_hyp):
                batch_bleu += sentence_bleu([ref], hyp)
            
            if batch_idx % print_every == 0:
                # Print batch_idx/num_batches and BLEU score
                print(f"Batch {batch_idx}/{total_step} - BLEU: {batch_bleu/len(batch_ref):.2f}")

            # Add batch BLEU score to total BLEU score
            total_bleu += batch_bleu

    # Calculate the average BLEU score and normalized loss over the entire validation or test set
    avg_bleu = total_bleu / len(data_loader.dataset)

    return avg_bleu
