import matplotlib.pyplot as plt
import seaborn as sns


def make_test_train_plot(history, title):
    sns.set()
    fig, axs = plt.subplots(2, figsize=(6, 8))

    x = list(range(0, len(history["binary_accuracy"])))

    fig.suptitle(title)
    axs[0].plot(x, history["binary_accuracy"], label="Training")
    axs[0].plot(x, history["val_binary_accuracy"], label="Validation")
    axs[1].plot(x, history["loss"], label="Training")
    axs[1].plot(x, history["val_loss"], label="Validation")

    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")

    axs[0].legend()
    axs[1].legend()

    return fig
