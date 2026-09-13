[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techaarvam/byom_workshop/blob/main/attention_ann_part2_puzzle_solution.ipynb)

# Sequence dependency: `disobeys`


```python
import torch

torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
```

### Residual layout

| idx | slot |
| --- | --- |
| 0 | `fly` |
| 1 | `speak` |
| 2 | `swap_fly` |
| 3 | `swap_speak` |
| 4 | `object` |
| 5 | `action_fly` |
| 6 | `action_speak` |
| 7 | `question` |
| 8 | `disobey` |
| 9 | `pos0` |
| 10 | `pos1` |
| 11 | `pos2` |
| 12 | `pos3` |
| 13 | `pos4` |
| 14 | `pos5` |
| 15 | `previous_word_is_disobey` |
| 16 | `object_attribute_fly` |
| 17 | `object_attribute_speak` |
| 18 | `is_swap_attr_fly` |
| 19 | `is_attr_fly_disobeyed` |
| 20 | `is_swap_attr_speak` |
| 21 | `is_attr_speak_disobeyed` |


```python
idx = {"fly": 0, "speak": 1, "swap_fly": 2, "swap_speak": 3,
       "object": 4, "action_fly": 5, "action_speak": 6, "question": 7, "disobey": 8}

position_start, L = 9, 6             # position one-hot occupies 9 .. 14
previous_word_is_disobey = 15
object_attribute_fly, object_attribute_speak = 16, 17
is_swap_attr_fly, is_attr_fly_disobeyed = 18, 19
is_swap_attr_speak, is_attr_speak_disobeyed = 20, 21
num_bits = 22

slot_names = {v: k for k, v in idx.items()}
slot_names.update({position_start + p: f"pos{p}" for p in range(L)})
slot_names.update({previous_word_is_disobey: "previous_word_is_disobey",
                   object_attribute_fly: "object_attribute_fly",
                   object_attribute_speak: "object_attribute_speak",
                   is_swap_attr_fly: "is_swap_attr_fly",
                   is_attr_fly_disobeyed: "is_attr_fly_disobeyed",
                   is_swap_attr_speak: "is_swap_attr_speak",
                   is_attr_speak_disobeyed: "is_attr_speak_disobeyed"})

for i in range(num_bits):
    print(f"{i:2} {slot_names[i]}")
```


     0 fly
     1 speak
     2 swap_fly
     3 swap_speak
     4 object
     5 action_fly
     6 action_speak
     7 question
     8 disobey
     9 pos0
    10 pos1
    11 pos2
    12 pos3
    13 pos4
    14 pos5
    15 previous_word_is_disobey
    16 object_attribute_fly
    17 object_attribute_speak
    18 is_swap_attr_fly
    19 is_attr_fly_disobeyed
    20 is_swap_attr_speak
    21 is_attr_speak_disobeyed


### Vocabulary


```python
content = {
    #                   fly spk swF swS  obj actF actS  q  dis
    "Rock":             [0,  0,  0,  0,   1,  0,  0,  0,  0],
    "Human":            [0,  1,  0,  0,   1,  0,  0,  0,  0],
    "Crow":             [1,  0,  0,  0,   1,  0,  0,  0,  0],
    "Flying superhero": [1,  1,  0,  0,   1,  0,  0,  0,  0],
    "swap-flight":      [0,  0,  1,  0,   0,  1,  0,  0,  0],
    "swap-speech":      [0,  0,  0,  1,   0,  0,  1,  0,  0],
    "keep-flight":      [0,  0,  0,  0,   0,  1,  0,  0,  0],
    "keep-speech":      [0,  0,  0,  0,   0,  0,  1,  0,  0],
    "disobeys":         [0,  0,  0,  0,   0,  0,  0,  0,  1],
    "he-is?":           [0,  0,  0,  0,   0,  0,  0,  1,  0],
}

for tok, bits in content.items():
    print(f"{tok:18} {torch.tensor(bits)}")
```


    Rock               tensor([0, 0, 0, 0, 1, 0, 0, 0, 0])
    Human              tensor([0, 1, 0, 0, 1, 0, 0, 0, 0])
    Crow               tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])
    Flying superhero   tensor([1, 1, 0, 0, 1, 0, 0, 0, 0])
    swap-flight        tensor([0, 0, 1, 0, 0, 1, 0, 0, 0])
    swap-speech        tensor([0, 0, 0, 1, 0, 0, 1, 0, 0])
    keep-flight        tensor([0, 0, 0, 0, 0, 1, 0, 0, 0])
    keep-speech        tensor([0, 0, 0, 0, 0, 0, 1, 0, 0])
    disobeys           tensor([0, 0, 0, 0, 0, 0, 0, 0, 1])
    he-is?             tensor([0, 0, 0, 0, 0, 0, 0, 1, 0])


$$x_p \;=\; \underbrace{c(t_p)}_{\text{9 content bits}} \;\Vert\; \underbrace{e_p}_{\text{6 position bits}} \;\Vert\; \underbrace{0}_{\text{7 scratch slots}}$$


```python
def embed(sentence):
    X = torch.zeros(len(sentence), num_bits)
    for p, tok in enumerate(sentence):
        X[p, :9] = torch.tensor(content[tok], dtype=torch.float32)
        X[p, position_start + p] = 1
    return X


sentence = ["Crow", "disobeys", "keep-flight", "swap-speech", "he-is?"]
X = embed(sentence)

for tok, row in zip(sentence, X):
    print(f"{tok:14} {row[:9]}  {row[position_start:position_start + L]}  {row[previous_word_is_disobey:]}")
```


    Crow           tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.])  tensor([1., 0., 0., 0., 0., 0.])  tensor([0., 0., 0., 0., 0., 0., 0.])
    disobeys       tensor([0., 0., 0., 0., 0., 0., 0., 0., 1.])  tensor([0., 1., 0., 0., 0., 0.])  tensor([0., 0., 0., 0., 0., 0., 0.])
    keep-flight    tensor([0., 0., 0., 0., 0., 1., 0., 0., 0.])  tensor([0., 0., 1., 0., 0., 0.])  tensor([0., 0., 0., 0., 0., 0., 0.])
    swap-speech    tensor([0., 0., 0., 1., 0., 0., 1., 0., 0.])  tensor([0., 0., 0., 1., 0., 0.])  tensor([0., 0., 0., 0., 0., 0., 0.])
    he-is?         tensor([0., 0., 0., 0., 0., 0., 0., 1., 0.])  tensor([0., 0., 0., 0., 1., 0.])  tensor([0., 0., 0., 0., 0., 0., 0.])


$$\operatorname{head}(X)=\operatorname{softmax}\!\left(\frac{(XW_Q)(XW_K)^{\top}}{\sqrt{d_h}}\right)XW_V$$


```python
def softmax(z, dim=-1):
    z = z - z.max(dim=dim, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=dim, keepdim=True)


def head(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = softmax(Q @ K.T / Wq.shape[1] ** 0.5)
    return A @ V, A
```

## Layer 1
### Head: Object Head


```python
Wq_O = torch.zeros(num_bits, 2); Wq_O[idx["question"]] = torch.tensor([8.0, 0.0])
Wk_O = torch.zeros(num_bits, 2); Wk_O[idx["object"]]   = torch.tensor([1.0, 0.0])
Wv_O = torch.zeros(num_bits, 2); Wv_O[idx["fly"]] = torch.tensor([1.0, 0.0]); Wv_O[idx["speak"]] = torch.tensor([0.0, 1.0])
Wo_O = torch.zeros(2, num_bits); Wo_O[0, object_attribute_fly] = 1; Wo_O[1, object_attribute_speak] = 1

print("Wq_O nonzero rows:", (Wq_O != 0).any(1).nonzero().flatten())
print("Wk_O nonzero rows:", (Wk_O != 0).any(1).nonzero().flatten())
print("Wv_O nonzero rows:", (Wv_O != 0).any(1).nonzero().flatten())
print("Wo_O nonzero cols:", (Wo_O != 0).any(0).nonzero().flatten())
```


    Wq_O nonzero rows: tensor([7])
    Wk_O nonzero rows: tensor([4])
    Wv_O nonzero rows: tensor([0, 1])
    Wo_O nonzero cols: tensor([16, 17])


### Head: Disobey Position Finder
$$W_Q^{(P)}=S\sum_{p=1}^{L-1} e_{\,\text{pos}_p}\,e_{p-1}^{\top}
\qquad
W_K^{(P)}=\sum_{p=0}^{L-1} e_{\,\text{pos}_p}\,e_{p}^{\top}$$

$$M=W_Q^{(P)}W_K^{(P)\top}\quad\Longrightarrow\quad q_i\!\cdot\!k_j = S\,[\,j=i-1\,]$$

$$M\neq M^{\top}$$


```python
S = 24.0

Wq_P = torch.zeros(num_bits, L)
for p in range(1, L):
    Wq_P[position_start + p, p - 1] = S

Wk_P = torch.zeros(num_bits, L)
for p in range(L):
    Wk_P[position_start + p, p] = 1

Wv_P = torch.zeros(num_bits, 1); Wv_P[idx["disobey"], 0] = 1
Wo_P = torch.zeros(1, num_bits); Wo_P[0, previous_word_is_disobey] = 1

M = Wq_P @ Wk_P.T
print("M[9:15, 9:15] =")
print(M[position_start:position_start + L, position_start:position_start + L])
print("symmetric:", torch.allclose(M, M.T))
```


    M[9:15, 9:15] =
    tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
            [24.,  0.,  0.,  0.,  0.,  0.],
            [ 0., 24.,  0.,  0.,  0.,  0.],
            [ 0.,  0., 24.,  0.,  0.,  0.],
            [ 0.,  0.,  0., 24.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 24.,  0.]])
    symmetric: False


$$\operatorname{FFN}_1(x)=0
\qquad
X_1 = X + \operatorname{head}_O(X)\,W_O^{(O)} + \operatorname{head}_P(X)\,W_O^{(P)}$$


```python
def layer1(X):
    oO, AO = head(X, Wq_O, Wk_O, Wv_O)
    oP, AP = head(X, Wq_P, Wk_P, Wv_P)
    X1 = X + oO @ Wo_O + oP @ Wo_P
    return X1, AO, AP


X1, AO, AP = layer1(X)

print("A_P")
print(AP)
print()
for tok, row in zip(sentence, X1):
    print(f"{tok:14} previous_word_is_disobey={row[previous_word_is_disobey]:.3f}   "
          f"object_attribute_fly={row[object_attribute_fly]:.3f}  "
          f"object_attribute_speak={row[object_attribute_speak]:.3f}")
```


    A_P
    tensor([[0.20, 0.20, 0.20, 0.20, 0.20],
            [1.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.00]])

    Crow           previous_word_is_disobey=0.200   object_attribute_fly=0.200  object_attribute_speak=0.000
    disobeys       previous_word_is_disobey=0.000   object_attribute_fly=0.200  object_attribute_speak=0.000
    keep-flight    previous_word_is_disobey=1.000   object_attribute_fly=0.200  object_attribute_speak=0.000
    swap-speech    previous_word_is_disobey=0.000   object_attribute_fly=0.200  object_attribute_speak=0.000
    he-is?         previous_word_is_disobey=0.000   object_attribute_fly=0.986  object_attribute_speak=0.000


## Layer 2
### Head: Get Flight Attributes and Head: Get Speech Attributes


```python
Wq_F = torch.zeros(num_bits, 1); Wq_F[idx["question"], 0] = 8
Wk_F = torch.zeros(num_bits, 1); Wk_F[idx["action_fly"], 0] = 1
Wv_F = torch.zeros(num_bits, 2); Wv_F[idx["swap_fly"]] = torch.tensor([1.0, 0.0]); Wv_F[previous_word_is_disobey] = torch.tensor([0.0, 1.0])

Wq_S = torch.zeros(num_bits, 1); Wq_S[idx["question"], 0] = 8
Wk_S = torch.zeros(num_bits, 1); Wk_S[idx["action_speak"], 0] = 1
Wv_S = torch.zeros(num_bits, 2); Wv_S[idx["swap_speak"]] = torch.tensor([1.0, 0.0]); Wv_S[previous_word_is_disobey] = torch.tensor([0.0, 1.0])

Wo_2 = torch.zeros(4, num_bits)
Wo_2[0, is_swap_attr_fly]        = 1
Wo_2[1, is_attr_fly_disobeyed]   = 1
Wo_2[2, is_swap_attr_speak]      = 1
Wo_2[3, is_attr_speak_disobeyed] = 1

print("Wv_F nonzero rows:", (Wv_F != 0).any(1).nonzero().flatten())
print("Wv_S nonzero rows:", (Wv_S != 0).any(1).nonzero().flatten())
print("Wo_2 nonzero cols:", (Wo_2 != 0).any(0).nonzero().flatten())
```


    Wv_F nonzero rows: tensor([ 2, 15])
    Wv_S nonzero rows: tensor([ 3, 15])
    Wo_2 nonzero cols: tensor([18, 19, 20, 21])


$$X_2 = X_1 + \big[\operatorname{head}_F(X_1)\;\Vert\;\operatorname{head}_S(X_1)\big]\,W_O^{(2)}$$


```python
def layer2_attn(X1):
    oF, AF = head(X1, Wq_F, Wk_F, Wv_F)
    oS, AS = head(X1, Wq_S, Wk_S, Wv_S)
    X2 = X1 + torch.cat([oF, oS], dim=1) @ Wo_2
    return X2, AF, AS


X2, AF, AS = layer2_attn(X1)
q = sentence.index("he-is?")

print("A_F[q]", AF[q])
print("A_S[q]", AS[q])
print()
print("is_swap_attr_fly       ", X2[q, is_swap_attr_fly])
print("is_attr_fly_disobeyed  ", X2[q, is_attr_fly_disobeyed])
print("is_swap_attr_speak     ", X2[q, is_swap_attr_speak])
print("is_attr_speak_disobeyed", X2[q, is_attr_speak_disobeyed])
```


    A_F[q] tensor([0.00, 0.00, 1.00, 0.00, 0.00])
    A_S[q] tensor([0.00, 0.00, 0.00, 1.00, 0.00])

    is_swap_attr_fly        tensor(0.)
    is_attr_fly_disobeyed   tensor(1.00)
    is_swap_attr_speak      tensor(1.00)
    is_attr_speak_disobeyed tensor(0.00)


### FFN 2

$$s_{\text{fly}} = x_{\text{object\_attribute\_fly}} + x_{\text{is\_swap\_attr\_fly}} + x_{\text{is\_attr\_fly\_disobeyed}}
\qquad
s_{\text{speak}} = x_{\text{object\_attribute\_speak}} + x_{\text{is\_swap\_attr\_speak}} + x_{\text{is\_attr\_speak\_disobeyed}}$$

$$\pi(s)=\operatorname{ReLU}(s)-2\operatorname{ReLU}(s-1)+2\operatorname{ReLU}(s-2)-2\operatorname{ReLU}(s-3)$$

$$\pi(0)=0,\quad \pi(1)=1,\quad \pi(2)=0,\quad \pi(3)=1$$


```python
W1 = torch.zeros(num_bits, 8)
b1 = torch.zeros(8)

for r in (object_attribute_fly, is_swap_attr_fly, is_attr_fly_disobeyed):
    W1[r, 0:4] = 1
for r in (object_attribute_speak, is_swap_attr_speak, is_attr_speak_disobeyed):
    W1[r, 4:8] = 1

b1[0:4] = torch.tensor([0.0, -1.0, -2.0, -3.0])
b1[4:8] = torch.tensor([0.0, -1.0, -2.0, -3.0])

W2 = torch.zeros(8, 2)
W2[0:4, 0] = torch.tensor([1.0, -2.0, 2.0, -2.0])
W2[4:8, 1] = torch.tensor([1.0, -2.0, 2.0, -2.0])

print("W1.T\n", W1.T)
print("\nb1", b1)
print("\nW2.T\n", W2.T)
print("\npi:", [float(torch.relu(torch.tensor([s, s - 1.0, s - 2.0, s - 3.0])) @ W2[0:4, 0]) for s in range(4)])
```


    W1.T
     tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.]])

    b1 tensor([ 0., -1., -2., -3.,  0., -1., -2., -3.])

    W2.T
     tensor([[ 1., -2.,  2., -2.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1., -2.,  2., -2.]])

    pi: [0.0, 1.0, 0.0, 1.0]


## Forward


```python
def forward(sentence):
    X = embed(sentence)
    X1, AO, AP = layer1(X)
    X2, AF, AS = layer2_attn(X1)
    H = torch.relu(X2 @ W1 + b1)
    Y = H @ W2
    return dict(X=X, X1=X1, X2=X2, Y=Y, AO=AO, AP=AP, AF=AF, AS=AS)


word_to_attributes = {
    "Rock":              (0, 0, 0),
    "Human":             (0, 0, 1),
    "Car":               (0, 1, 0),
    "Talking Tow Truck": (0, 1, 1),
    "Crow":              (1, 0, 0),
    "Flying Superhero":  (1, 0, 1),
    "Plane":             (1, 1, 0),
    "Talking Planes":    (1, 1, 1),
}
attributes_to_word = {b: w for w, b in word_to_attributes.items()}


def readout(sentence):
    r = forward(sentence)
    q = sentence.index("he-is?")
    fly, speak = (int(v.round()) for v in r["Y"][q])
    return attributes_to_word[(fly, 0, speak)], r["Y"][q]
```


```python
sentences = [
    ["Human", "disobeys", "keep-flight", "disobeys", "swap-speech", "he-is?"],
    ["Crow", "disobeys", "keep-flight", "swap-speech", "he-is?"],
    ["Crow", "keep-flight", "disobeys", "swap-speech", "he-is?"],
    ["Crow", "disobeys", "keep-flight", "disobeys", "swap-speech", "he-is?"],
]

for s in sentences:
    word, y = readout(s)
    print(f"{' '.join(s):58} {y}  ->  {word}")
```


    Human disobeys keep-flight disobeys swap-speech he-is?     tensor([1.00, 0.98])  ->  Flying Superhero
    Crow disobeys keep-flight swap-speech he-is?               tensor([0.02, 1.00])  ->  Human
    Crow keep-flight disobeys swap-speech he-is?               tensor([0.99, 0.00])  ->  Crow
    Crow disobeys keep-flight disobeys swap-speech he-is?      tensor([0.02, 0.00])  ->  Rock


$$\{\text{Crow},\,\text{disobeys},\,\text{keep-flight},\,\text{swap-speech},\,\text{he-is?}\}$$


```python
a = ["Crow", "disobeys", "keep-flight", "swap-speech", "he-is?"]
b = ["Crow", "keep-flight", "disobeys", "swap-speech", "he-is?"]

print(sorted(a) == sorted(b))
print(readout(a)[0])
print(readout(b)[0])
```


    True
    Human
    Crow



```python
for s in sentences:
    r = forward(s)
    print(" ".join(s))
    print("  A_P")
    for tok, row in zip(s, r["AP"]):
        print(f"    {tok:14} {row}")
    print("  previous_word_is_disobey", r["X1"][:, previous_word_is_disobey])
    q = s.index("he-is?")
    print("  gathered  ", r["X2"][q, [is_swap_attr_fly, is_attr_fly_disobeyed,
                                      is_swap_attr_speak, is_attr_speak_disobeyed]])
    print("  Y         ", r["Y"][q])
    print()
```


    Human disobeys keep-flight disobeys swap-speech he-is?
      A_P
        Human          tensor([0.17, 0.17, 0.17, 0.17, 0.17, 0.17])
        disobeys       tensor([1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        keep-flight    tensor([0.00, 1.00, 0.00, 0.00, 0.00, 0.00])
        disobeys       tensor([0.00, 0.00, 1.00, 0.00, 0.00, 0.00])
        swap-speech    tensor([0.00, 0.00, 0.00, 1.00, 0.00, 0.00])
        he-is?         tensor([0.00, 0.00, 0.00, 0.00, 1.00, 0.00])
      previous_word_is_disobey tensor([0.33, 0.00, 1.00, 0.00, 1.00, 0.00])
      gathered   tensor([0.00, 1.00, 1.00, 1.00])
      Y          tensor([1.00, 0.98])

    Crow disobeys keep-flight swap-speech he-is?
      A_P
        Crow           tensor([0.20, 0.20, 0.20, 0.20, 0.20])
        disobeys       tensor([1.00, 0.00, 0.00, 0.00, 0.00])
        keep-flight    tensor([0.00, 1.00, 0.00, 0.00, 0.00])
        swap-speech    tensor([0.00, 0.00, 1.00, 0.00, 0.00])
        he-is?         tensor([0.00, 0.00, 0.00, 1.00, 0.00])
      previous_word_is_disobey tensor([0.20, 0.00, 1.00, 0.00, 0.00])
      gathered   tensor([0.00, 1.00, 1.00, 0.00])
      Y          tensor([0.02, 1.00])

    Crow keep-flight disobeys swap-speech he-is?
      A_P
        Crow           tensor([0.20, 0.20, 0.20, 0.20, 0.20])
        keep-flight    tensor([1.00, 0.00, 0.00, 0.00, 0.00])
        disobeys       tensor([0.00, 1.00, 0.00, 0.00, 0.00])
        swap-speech    tensor([0.00, 0.00, 1.00, 0.00, 0.00])
        he-is?         tensor([0.00, 0.00, 0.00, 1.00, 0.00])
      previous_word_is_disobey tensor([0.20, 0.00, 0.00, 1.00, 0.00])
      gathered   tensor([0.00, 0.00, 1.00, 1.00])
      Y          tensor([0.99, 0.00])

    Crow disobeys keep-flight disobeys swap-speech he-is?
      A_P
        Crow           tensor([0.17, 0.17, 0.17, 0.17, 0.17, 0.17])
        disobeys       tensor([1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        keep-flight    tensor([0.00, 1.00, 0.00, 0.00, 0.00, 0.00])
        disobeys       tensor([0.00, 0.00, 1.00, 0.00, 0.00, 0.00])
        swap-speech    tensor([0.00, 0.00, 0.00, 1.00, 0.00, 0.00])
        he-is?         tensor([0.00, 0.00, 0.00, 0.00, 1.00, 0.00])
      previous_word_is_disobey tensor([0.33, 0.00, 1.00, 0.00, 1.00, 0.00])
      gathered   tensor([0.00, 1.00, 1.00, 1.00])
      Y          tensor([0.02, 0.00])



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
