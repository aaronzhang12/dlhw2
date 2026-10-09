from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from cnn import CNN
from mlp import MLP

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    num_inputs = train_inputs.shape[0]
    
    shuffled_indices = tf.random.shuffle(tf.range(num_inputs))
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)

    for i in range(math.ceil(num_inputs/model.batch_size)):
        # start training epochs over all sequential batches
        batch_inputs, batch_labels = get_next_batch(i, shuffled_inputs, shuffled_labels, model.batch_size)

        with tf.GradientTape() as tape:
            # Computes the gradients of all trainable vars w.r.t loss
            logits = model(batch_inputs)
            loss = model.loss(logits, batch_labels)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        # Adjusts the trainable vars according to the optimizer update rule
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: 
        test accuracy - the fraction of correctly classified test inputs
        test logits - shape (num_inputs, num_classes), in input order
    """
    num_inputs = test_inputs.shape[0]
    batch_logits = []

    for i in range(math.ceil(num_inputs / model.batch_size)):
        batch_inputs, _ = get_next_batch(
            i, test_inputs, test_labels, model.batch_size
        )
        batch_logits.append(model(batch_inputs, is_testing=True))

    logits = tf.concat(batch_logits, axis=0)
    accuracy = model.accuracy(logits, test_labels)
    return (accuracy, logits)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, logits, image_labels, first_label, second_label, third_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "deer"
    :param third_label: the name of the third class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            predicted_index = predicted_labels[ind]
            actual_index = np.argmax(image_labels[ind], axis=0)
            labels = [first_label, second_label, third_label]
            pl = labels[predicted_index]
            al = labels[actual_index]
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def main():
    '''
    Read in CIFAR10 data (limited to 3 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    Students should receive a final accuracy 
    on the testing examples for cat, deer and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = "/Users/aaronzhang/Desktop/cs2470/HW2-CNN-F26-Stencil/data/train"
    LOCAL_TEST_FILE = "/Users/aaronzhang/Desktop/cs2470/HW2-CNN-F26-Stencil/data/test"

    # for mapping one-hot class back to original
    classes = [3,4,5]
    class_ids = np.asarray(sorted(classes), dtype=np.int64)

    # Load your testing and training data using the get_data function
    train_data, train_labels = get_data(AUTOGRADER_TRAIN_FILE, classes)
    test_data, test_labels = get_data(AUTOGRADER_TEST_FILE, classes)

    # initialize model and optimizer
    mlp = MLP(classes)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0003)

    for epoch in range(20):
        train(mlp, optimizer, train_data, train_labels)

        # get the current training and test accuracy at each epoch to test overfitting
        train_acc, train_logits = test(mlp, train_data, train_labels)
        test_acc, _ = test(mlp, test_data, test_labels)
        train_loss = mlp.loss(train_logits, train_labels)

        print(
            f"MLP Epoch {epoch + 1}: loss={float(train_loss):.3f}, "
            f"train={float(train_acc):.1%}, test={float(test_acc):.1%}"
        )

    accuracy, predictions = test(mlp, test_data, test_labels)
    np.save("predictions_mlp.npy", class_ids[np.argmax(predictions.numpy(), axis=1)])
    print(accuracy)
    # cnn = CNN(classes)

    # for epoch in range(15):
    #     train(cnn, optimizer, train_data, train_labels)

    #     # get the current training and test accuracy at each epoch to test overfitting
    #     train_acc, train_logits = test(cnn, train_data, train_labels)
    #     test_acc, _ = test(cnn, test_data, test_labels)
    #     train_loss = cnn.loss(train_logits, train_labels)

    #     print(
    #         f"CNN Epoch {epoch + 1}: loss={float(train_loss):.3f}, "
    #         f"train={float(train_acc):.1%}, test={float(test_acc):.1%}"
    #     )

    # accuracy, predictions = test(cnn, test_data, test_labels)
    # print(accuracy)

    # predicted_class_ids = class_ids[np.argmax(predictions.numpy(), axis=1)]
    # np.save("predictions_cnn.npy", predicted_class_ids)
    # return

if __name__ == '__main__':
    main()
