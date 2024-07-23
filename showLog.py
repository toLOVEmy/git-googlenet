import matplotlib.pyplot as plt
import pandas as pd

def plot_training_log(log_file):
    # Read the log file
    log_data = pd.read_csv(log_file)

    epochs = log_data['Epoch']
    train_loss = log_data['Training Loss']
    val_loss = log_data['Validation Loss']
    train_acc = log_data['Training Accuracy']
    val_acc = log_data['Validation Accuracy']
    time_taken = log_data['Time Taken']

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Plot Time Taken per Epoch
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, time_taken, label='Time Taken (s)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Time Taken (seconds)')
    # plt.title('Time Taken per Epoch')
    # plt.legend()
    #
    # plt.show()

if __name__ == '__main__':
    log_file = 'training_log012.txt'
    plot_training_log(log_file)
