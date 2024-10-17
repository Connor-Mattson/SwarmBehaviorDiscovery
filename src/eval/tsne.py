# Supress the FutureWarning in the TSNE class
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

def embed_tsne(X, labels, path_or_disp=None, is_3d=False, lr="auto", init="pca", perplexity=12, early_exaggeration=5):
    n_components = 3 if is_3d else 2
    reduced = TSNE(
        n_components=n_components,
        learning_rate=lr,
        init=init,
        # perplexity=perplexity,
        # early_exaggeration=early_exaggeration
    ).fit_transform(X)

    # color_classes = {
    #     -1: [0.33, 0.33, 0.33],
    #     0: [0.8, 0.8, 0.8],
    #     1: [0, 1, 0],
    #     2: [0, 0, 1],
    #     3: [0, 1, 1],  # Cyan
    #     4: [1, 1, 0],  # Yellow
    #     5: [1.0, 0, 0.7]  # Pink
    # }

    color_classes = {
        -1: [0.33, 0.33, 0.33],
        0: [0.8, 0.8, 0.8],
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",  # Cyan
        4: "tab:purple",  # Yellow
        5: "tab:blue",  # Pink
    }

    label_classes = {
        # -1: "Unlabeled",
        0: "Random",
        1: "Cyclic Pursuit",
        2: "Aggregation",
        3: "Dispersal",
        4: "Milling",
        5: "Egg"
    }
    fig = plt.figure()
    for key in label_classes:
        # t1 = [reduced[i][0] for i in range(len(reduced))]
        # t2 = [(labels[j] == key) for j in range(len(reduced))]
        # print(t2)
        x = [reduced[i][0] for i in range(len(reduced)) if (labels[i] == key)]
        y = [reduced[i][1] for i in range(len(reduced)) if (labels[i] == key)]
        colors = [color_classes[labels[i]] for i in range(len(reduced)) if (labels[i] == key)]
        legend = [label_classes[labels[i]] for i in range(len(reduced)) if (labels[i] == key)]
        plt.grid(False)
        if x:
            plt.scatter(x, y, c=colors, label=legend[key])

    plt.legend(loc="best")
    # plt.title("t-SNE embedding of behavior features")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    # plt.show()
    plt.savefig(f"tsne_{int(time.time())}.pdf")