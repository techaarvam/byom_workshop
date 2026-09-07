[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techaarvam/byom_workshop/blob/main/attention_ann.ipynb)

*Part of the TechAarvam workshop support files — [Build Your Own Model](https://www.techaarvam.com/workshops/build-your-own-model).*

# Attention is simpler than you think
### A hand-crafted superhero transformer

This notebook presents a hand-crafted example - that presents the intuition behind the attention block in the transformer architecture.

Attention and FFN are the two main components. FFN is an Artificial neural network (ANN) with 1 input, 1 hidden , 1 output layers. 

So we start constructing the FFN's job by hand.
Then show how Attention extracts what that FFN is able to use from a sentence.
The sentences used and the vocabulary are intentionally hand designed to be simple. The attention block weights are hand-coded, instead of trained.
The ANN(FFN) is trained. 


```python
import numpy as np

np.set_printoptions(precision=2, suppress=True)
```

---
## Part 1: Artificial Neural Network

### Vocabulary for the ANN

| Word | can fly? | has wheels? | can speak? |
| :--- | :---: | :---: | :---: |
| Rock | 0 | 0 | 0 |
| Human | 0 | 0 | 1 |
| Car | 0 | 1 | 0 |
| Talking Tow Truck | 0 | 1 | 1 |
| Crow | 1 | 0 | 0 |
| Flying Superhero | 1 | 0 | 1 |
| Plane | 1 | 1 | 0 |
| Talking Planes | 1 | 1 | 1 |


```python
ANN_VOCAB = {
    "Rock":              (0, 0, 0),
    "Human":             (0, 0, 1),
    "Car":               (0, 1, 0),
    "Talking Tow Truck": (0, 1, 1),
    "Crow":              (1, 0, 0),
    "Flying Superhero":  (1, 0, 1),
    "Plane":             (1, 1, 0),
    "Talking Planes":    (1, 1, 1),
}

for word, bits in ANN_VOCAB.items():
    print(f"{word:20} {bits}")
```

    Rock                 (0, 0, 0)
    Human                (0, 0, 1)
    Car                  (0, 1, 0)
    Talking Tow Truck    (0, 1, 1)
    Crow                 (1, 0, 0)
    Flying Superhero     (1, 0, 1)
    Plane                (1, 1, 0)
    Talking Planes       (1, 1, 1)


### ANN's learning goal

The ANN's output - the logits (probability scores) for the next word is a correction to the input word.

The input is fixed bit-encoding of 6-bits. No sentences/tokens/words yet.

    Input word  - 3 bits, one-hot, or IDs (design choice)
    Correction  - 3 bits
    Output word - attributes after the correction

### Example

1. Rock + add flight -> Plane
2. Human + add flight -> Flying Superhero


```python
BY_BITS = {bits: word for word, bits in ANN_VOCAB.items()}


def apply_correction(word, correction):
    """Correction is 3 bits: 1 means flip that attribute."""
    bits = ANN_VOCAB[word]
    out = tuple(b ^ c for b, c in zip(bits, correction))
    return BY_BITS[out]


print(apply_correction("Rock", (1, 0, 0)))   # add flight
print(apply_correction("Human", (1, 0, 0)))  # add flight
```

    Crow
    Flying Superhero


### Training the ANN: Pseudocode


```python
# Pseudocode - not run here.
#
# for loop in range (num_epochs):
#     for batch_inputs, batch_targets in loader:
#         prediction = model.forward(batch_inputs)          # predict
#         loss = CrossEntropyLoss(prediction, target)       # how far off?
#         loss.backward()                                   # gradient descent
```

In the full workshop, the notebook with the full ANN implementation is available. Visit the relevant TechAarvam pages for locating them. 

---
## Part 2: Attention Block

Input is now a sentence:

> **Crow keep-flight swap-speech he-is?**

Sentences have complex structure. Word order varies.
So: fixed sentence structure, small defined vocabulary.

    Objects:       Rock, Human, Crow, Flying superhero
    Actions:       swap-flight, swap-speech, keep-flight, keep-speech
    Interrogative: he-is?

### Tokens to vectors

8-bit vectors, shown as 4 + 4.

**Bits 1-4 - attributes and correction:**

| Token | can fly | can speak | swap flight | swap speech |
| :--- | :---: | :---: | :---: | :---: |
| Rock | 0 | 0 | 0 | 0 |
| Human | 0 | 1 | 0 | 0 |
| Crow | 1 | 0 | 0 | 0 |
| Flying superhero | 1 | 1 | 0 | 0 |
| swap-flight | 0 | 0 | 1 | 0 |
| swap-speech | 0 | 0 | 0 | 1 |
| keep-flight | 0 | 0 | 0 | 0 |
| keep-speech | 0 | 0 | 0 | 0 |
| he-is? | 0 | 0 | 0 | 0 |

**Bits 5-8 - token type:**

| Token | object? | flight act? | speech act? | question? |
| :--- | :---: | :---: | :---: | :---: |
| Rock | 1 | 0 | 0 | 0 |
| Human | 1 | 0 | 0 | 0 |
| Crow | 1 | 0 | 0 | 0 |
| Flying superhero | 1 | 0 | 0 | 0 |
| swap-flight | 0 | 1 | 0 | 0 |
| swap-speech | 0 | 0 | 1 | 0 |
| keep-flight | 0 | 1 | 0 | 0 |
| keep-speech | 0 | 0 | 1 | 0 |
| he-is? | 0 | 0 | 0 | 1 |


```python
BITS = ["can fly", "can speak", "swap flight", "swap speech",
        "object?", "flight act?", "speech act?", "question?"]

VOCAB = {
    #                    fly spk swF swS  obj flA spA  q
    "Rock":             [0,  0,  0,  0,   1,  0,  0,  0],
    "Human":            [0,  1,  0,  0,   1,  0,  0,  0],
    "Crow":             [1,  0,  0,  0,   1,  0,  0,  0],
    "Flying superhero": [1,  1,  0,  0,   1,  0,  0,  0],
    "swap-flight":      [0,  0,  1,  0,   0,  1,  0,  0],
    "swap-speech":      [0,  0,  0,  1,   0,  0,  1,  0],
    "keep-flight":      [0,  0,  0,  0,   0,  1,  0,  0],
    "keep-speech":      [0,  0,  0,  0,   0,  0,  1,  0],
    "he-is?":           [0,  0,  0,  0,   0,  0,  0,  1],
}

VOCAB = {k: np.array(v) for k, v in VOCAB.items()}

for tok, v in VOCAB.items():
    print(f"{tok:18} {v[:4]}  {v[4:]}")
```

    Rock               [0 0 0 0]  [1 0 0 0]
    Human              [0 1 0 0]  [1 0 0 0]
    Crow               [1 0 0 0]  [1 0 0 0]
    Flying superhero   [1 1 0 0]  [1 0 0 0]
    swap-flight        [0 0 1 0]  [0 1 0 0]
    swap-speech        [0 0 0 1]  [0 0 1 0]
    keep-flight        [0 0 0 0]  [0 1 0 0]
    keep-speech        [0 0 0 0]  [0 0 1 0]
    he-is?             [0 0 0 0]  [0 0 0 1]


### The sentence


```python
SENTENCE = ["Crow", "keep-flight", "swap-speech", "he-is?"]

X = np.stack([VOCAB[t] for t in SENTENCE])
print(X.shape)
X
```

    (4, 8)





    array([[1, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1]])



### Attention Math

Skip the equations and go to the hand-written Q, K, V weights to get the idea behind these equations first. 

N heads. Each head does:

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_h}}+M\right)V$$

- $QK^T$ - pairwise scores
- softmax - normalize
- $V$ - send the payload


```python
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def head(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_h = Wq.shape[1]
    scores = Q @ K.T / np.sqrt(d_h)
    A = softmax(scores)
    return A @ V, A, Q, K, V
```

### Hand-written heads

Q - Query. K - Key. V - Value (payload).

First each head arrives and scores and based on the scores, ships the payload (V) as the output. We want to understand what the scores are first.

Scores: every pair of words gets a score of its relevance for this head. Attention block can have many heads.

Every word asks a question; And also the question is not the same across heads. Each head's each word can ask a different question; We have two heads in this hand-constructed example (Object Head, Action Verb-Object head)
Object head - `he-is?` asks: are you an Object?

Keys answer. Every token answers.
Object head - "Crow" replies: I am an Object.

### Object Head - weights

- Row 8 -> query `he-is?` in the object head asks - are you an object?
- Row 5 -> key replies to this question. (i.e every token replies)
- Rows 1-2 -> payload is can fly, can speak. (Every token can have a payload, and tokens with high-scores get to send their payload)


```python
Wq1 = np.zeros((8, 2)); Wq1[7] = [8, 0]
Wk1 = np.zeros((8, 2)); Wk1[4] = [1, 0]
Wv1 = np.zeros((8, 2)); Wv1[0] = [1, 0]; Wv1[1] = [0, 1]

print("Wq1\n", Wq1, "\n\nWk1\n", Wk1, "\n\nWv1\n", Wv1)
```

    Wq1
     [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [8. 0.]] 
    
    Wk1
     [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [1. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]] 
    
    Wv1
     [[1. 0.]
     [0. 1.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]]


### Action (Verb-Object) Head - weights

- Row 8 -> `he-is?` queries both slots
- Rows 6-7 -> keys are the two action types
- Rows 3-4 -> payload is swap flight, swap speech


```python
Wq2 = np.zeros((8, 2)); Wq2[7] = [8, 8]
Wk2 = np.zeros((8, 2)); Wk2[5] = [1, 0]; Wk2[6] = [0, 1]
Wv2 = np.zeros((8, 2)); Wv2[2] = [1, 0]; Wv2[3] = [0, 1]

print("Wq2\n", Wq2, "\n\nWk2\n", Wk2, "\n\nWv2\n", Wv2)
```

    Wq2
     [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [8. 8.]] 
    
    Wk2
     [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [1. 0.]
     [0. 1.]
     [0. 0.]] 
    
    Wv2
     [[0. 0.]
     [0. 0.]
     [1. 0.]
     [0. 1.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]]


### Object head: Derive Q, K, V from the weights:

Crow, bits 1-8: `1 0 0 0 1 0 0 0`

The payload is Crow's object attributes. Nothing else.


```python
crow = VOCAB["Crow"]

print("x_Crow           ", crow)
print("x_Crow @ Wk1     ", crow @ Wk1, "  <- key: I am an object")
print("x_Crow @ Wv1     ", crow @ Wv1, "  <- value: [can fly, can speak]")
```

    x_Crow            [1 0 0 0 1 0 0 0]
    x_Crow @ Wk1      [1. 0.]   <- key: I am an object
    x_Crow @ Wv1      [1. 0.]   <- value: [can fly, can speak]


### Multiply it out: swap-flight

swap-flight, bits 1-8: `0 0 1 0 0 1 0 0`

The payload is the correction. Nothing else.


```python
sf = VOCAB["swap-flight"]

print("x_swap-flight        ", sf)
print("x_swap-flight @ Wk2  ", sf @ Wk2, "  <- key: I am a flight action")
print("x_swap-flight @ Wv2  ", sf @ Wv2, "  <- value: [swap flight, swap speech]")
```

    x_swap-flight         [0 0 1 0 0 1 0 0]
    x_swap-flight @ Wk2   [1. 0.]   <- key: I am a flight action
    x_swap-flight @ Wv2   [1. 0.]   <- value: [swap flight, swap speech]


### he-is? collects both

`he-is?` is the last token, so we read row 3 of the attention matrix.


```python
out1, A1, Q1, K1, V1 = head(X, Wq1, Wk1, Wv1)
out2, A2, Q2, K2, V2 = head(X, Wq2, Wk2, Wv2)

q = SENTENCE.index("he-is?")

print("Object head - attention from he-is?")
for tok, k, a in zip(SENTENCE, K1, A1[q]):
    print(f"  {tok:14} key={k}  softmax={a:.2f}")
print("  head 1 output:", out1[q])

print("\nAction head - attention from he-is?")
for tok, k, a in zip(SENTENCE, K2, A2[q]):
    print(f"  {tok:14} key={k}  softmax={a:.2f}")
print("  head 2 output:", out2[q])
```

    Object head - attention from he-is?
      Crow           key=[1. 0.]  softmax=0.99
      keep-flight    key=[0. 0.]  softmax=0.00
      swap-speech    key=[0. 0.]  softmax=0.00
      he-is?         key=[0. 0.]  softmax=0.00
      head 1 output: [0.99 0.  ]
    
    Action head - attention from he-is?
      Crow           key=[0. 0.]  softmax=0.00
      keep-flight    key=[1. 0.]  softmax=0.50
      swap-speech    key=[0. 1.]  softmax=0.50
      he-is?         key=[0. 0.]  softmax=0.00
      head 2 output: [0.  0.5]


Two action words split the mass, so head 2 gives `[0, 0.5]`.
$W_O$ scales by 2 -> `[0, 1]`.

### Concat: 4 bits out

$$\underbrace{[\,1,\ 0\,]}_{\text{head 1}}\ \Vert\ \underbrace{[\,0,\ 1\,]}_{\text{head 2}}=[\,1,\ 0,\ 0,\ 1\,]$$


```python
Wo = np.diag([1.0, 1.0, 2.0, 2.0])   # head 2 mass was split across 2 words

concat = np.concatenate([out1[q], out2[q]])
final = concat @ Wo

print("concat        ", concat)
print("after W_O     ", final)
print()
for name, val in zip(BITS[:4], final):
    print(f"  {name:12} {val:.0f}")
```

    concat         [0.99 0.   0.   0.5 ]
    after W_O      [0.99 0.   0.   1.  ]
    
      can fly      1
      can speak    0
      swap flight  0
      swap speech  1


Hand-built weights pulled the object attributes and the
correction attributes into 4 bits.

Crow: flies, no speech. Correction: swap speech.
-> Flying superhero. Same 4 bits the ANN wanted.


```python
fly, speak, swap_fly, swap_speak = (int(v) for v in final.round())

obj_in = (fly, 0, speak)                       # ANN bits: fly, wheels, speak
correction = (swap_fly, 0, swap_speak)

print("object in :", BY_BITS[obj_in])
print("correction:", correction)
print("object out:", apply_correction(BY_BITS[obj_in], correction))
```

    object in : Crow
    correction: (0, 0, 1)
    object out: Flying Superhero


---

### About this file

This notebook is part of the support files for the TechAarvam workshop
**[Build Your Own Model](https://www.techaarvam.com/workshops/build-your-own-model)**.

- Website: <https://www.techaarvam.com>
- Workshop files repository: <https://github.com/techaarvam/byom_workshop>
- YouTube: <https://www.youtube.com/@TechAarvam>

© TechAarvam. You are free to use, copy, modify, share and build on this
material, including for commercial purposes, **provided you credit
TechAarvam** and link back to <https://www.techaarvam.com>. Please keep
this notice with any copy or derivative. Provided as-is, without warranty.
