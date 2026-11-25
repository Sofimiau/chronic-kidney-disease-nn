"""
Our neural network (MLP), built from scratch with just NumPy.

We are not allowed to use sklearn/pytorch/etc for the actual learning
algorithm, so everything here -- forward pass, backprop, and the weight
updates -- is written by hand following what we saw in class.

The network has ReLU in the hidden layers and a sigmoid at the output,
since this is a binary classification problem (ckd vs notckd). We train
it with mini-batch gradient descent and stop early when the validation
loss stops improving, so it doesn't just memorize the training set.
"""
import numpy as np


def sigmoid(z):
    # clip so we don't get overflow warnings from exp() when z is very
    # negative/positive
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    # derivative of relu: 1 if z was positive, 0 otherwise.
    # we need this during backprop, not forward.
    return (z > 0).astype(z.dtype)


class MLP:
    """A simple feed-forward neural network with one or more hidden layers.

    layer_sizes is a list like [24, 16, 8, 1]: 24 inputs (our features),
    two hidden layers (16 and 8 neurons) and 1 output neuron since we're
    doing binary classification.

    optimizer can be "sgd", "momentum" or "adam" -- we implemented all
    three so we could compare them in the hyperparameter search.
    """

    def __init__(self, layer_sizes, seed=42, optimizer="adam", lr=0.01,
                 l2=0.0, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of layers that have weights
        self.optimizer = optimizer
        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        rng = np.random.default_rng(seed)

        # W[i] and b[i] are the weights/bias going from layer i to layer i+1.
        # We keep them as a list, one matrix per layer, instead of one big
        # vector, because it matches how we think about the network and
        # makes the forward/backward loops easy to write.
        self.W = []
        self.b = []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            # He initialization: works better than plain random weights
            # when using ReLU, otherwise activations can explode or vanish
            # as they go through several layers.
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, std, size=(layer_sizes[i], layer_sizes[i + 1])))
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        # these keep track of past gradients, used by momentum and adam.
        # for plain sgd they just stay at zero and are never used.
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.t = 0  # counts how many updates we've done (needed for adam)

    def forward(self, X):
        """Runs the input through every layer and returns everything we
        computed along the way (we need it later for backprop).

        A holds the activations (A[0] is just the input X), Z holds the
        values before the activation function is applied.
        """
        A = [X]
        Z = []
        for i in range(self.L):
            z = A[-1] @ self.W[i] + self.b[i]
            Z.append(z)
            if i == self.L - 1:
                a = sigmoid(z)       # last layer -> probability between 0 and 1
            else:
                a = relu(z)          # hidden layers
            A.append(a)
        return A, Z

    def compute_loss(self, y_hat, y, eps=1e-10):
        """Binary cross-entropy, plus the L2 penalty if we're using one.

        eps is just there so we never take log(0), which would give us
        -inf / nan if a prediction is exactly 0 or 1.
        """
        m = y.shape[0]
        bce = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        l2_term = 0.0
        if self.l2 > 0:
            l2_term = (self.l2 / (2 * m)) * sum(np.sum(w ** 2) for w in self.W)
        return bce + l2_term

    def backward(self, A, Z, y):
        """This is backpropagation: we go from the output layer back to the
        input, using the chain rule to figure out how much each weight
        contributed to the error.

        For the output layer, because we're using sigmoid + cross-entropy
        together, the math simplifies nicely and the gradient is just
        (predicted - actual). That's the "dz = A[-1] - y" line below -- it
        looks too simple, but it's correct, it's a known result from
        combining those two functions.

        For every earlier layer we take the error signal from the layer in
        front of it, push it back through that layer's weights, and
        multiply by the relu derivative (since relu "blocks" the signal
        wherever the pre-activation was negative).
        """
        m = y.shape[0]
        dW = [None] * self.L
        db = [None] * self.L

        dz = A[-1] - y
        for i in reversed(range(self.L)):
            dW[i] = (A[i].T @ dz) / m + (self.l2 / m) * self.W[i]
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                da_prev = dz @ self.W[i].T
                dz = da_prev * relu_grad(Z[i - 1])  # chain rule through relu
        return dW, db

    def step(self, dW, db):
        """Updates the weights using whichever optimizer we picked.

        - momentum: keeps a running average of past gradients (like a ball
          rolling downhill that builds up speed), which helps it not get
          stuck oscillating.
        - adam: the fancier one, keeps a running average of the gradient
          AND of the gradient squared, and uses both to adapt the step
          size for each weight individually. This is the one we used for
          most of our final runs since it converged faster in testing.
        - anything else: plain gradient descent, just subtract lr * gradient.
        """
        self.t += 1
        if self.optimizer == "momentum":
            for i in range(self.L):
                self.vW[i] = self.momentum * self.vW[i] - self.lr * dW[i]
                self.vb[i] = self.momentum * self.vb[i] - self.lr * db[i]
                self.W[i] += self.vW[i]
                self.b[i] += self.vb[i]
        elif self.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i in range(self.L):
                self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
                self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
                self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
                self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
                # bias correction -- without this the first few updates are
                # too small, since mW/vW start at zero
                mW_hat = self.mW[i] / (1 - beta1 ** self.t)
                mb_hat = self.mb[i] / (1 - beta1 ** self.t)
                vW_hat = self.vW[i] / (1 - beta2 ** self.t)
                vb_hat = self.vb[i] / (1 - beta2 ** self.t)
                self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
        else:  # plain sgd
            for i in range(self.L):
                self.W[i] -= self.lr * dW[i]
                self.b[i] -= self.lr * db[i]

    def predict_proba(self, X):
        """Probability of being ckd for each row in X."""
        A, _ = self.forward(X)
        return A[-1]

    def predict(self, X, threshold=0.5):
        """0/1 prediction, just the probability rounded at 0.5."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32,
            patience=40, verbose=False, seed=0):
        """Trains the network.

        Every epoch we shuffle the training data and go through it in
        small batches, doing forward -> backward -> step for each one.
        After each epoch we check the loss/accuracy on train and val so we
        can plot it later and see if the model is overfitting.

        Early stopping: we save a copy of the weights every time val loss
        gets better. If it doesn't improve for "patience" epochs in a row,
        we stop and go back to the best copy we saved, instead of keeping
        whatever the last epoch happened to be (which could be worse if it
        started overfitting).
        """
        rng = np.random.default_rng(seed)
        n = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = np.inf
        best_weights = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # shuffle so each epoch sees the data in a different order
            perm = rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                A, Z = self.forward(xb)
                dW, db = self.backward(A, Z, yb)
                self.step(dW, db)

            # check performance on the whole train/val set (not just the
            # last batch) so the curves we plot are smooth
            train_proba = self.predict_proba(X_train)
            val_proba = self.predict_proba(X_val)
            train_loss = self.compute_loss(train_proba, y_train)
            val_loss = self.compute_loss(val_proba, y_val)
            train_acc = np.mean((train_proba >= 0.5) == y_train)
            val_acc = np.mean((val_proba >= 0.5) == y_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_weights = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and epoch % 50 == 0:
                print(f"epoch {epoch:4d}  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

        # go back to the weights that had the best val loss, not the last ones
        if best_weights is not None:
            self.W, self.b = best_weights
        history["stopped_epoch"] = epoch
        history["best_val_loss"] = best_val_loss
        return history
