# Build Your Own Model — Workshop Files

Miscellaneous support files for the TechAarvam workshop
**[Build Your Own Model](https://www.techaarvam.com/workshops/build-your-own-model)**.

This is a grab-bag repository: notebooks, scripts and small examples that
accompany the workshop sessions and the videos. It is not a library and has no
stable API — files are added and updated as the workshop material evolves.

## Links

- Workshop: <https://www.techaarvam.com/workshops/build-your-own-model>
- YouTube: <https://www.youtube.com/@TechAarvam>
- Website: <https://www.techaarvam.com>

## Contents

| File | What it is |
| :--- | :--- |
| [`attention_ann.ipynb`](attention_ann.ipynb) | A hand-crafted transformer: attention weights written by hand, and the ANN (FFN) whose job they feed. [Open in Colab](https://colab.research.google.com/github/techaarvam/byom_workshop/blob/main/attention_ann.ipynb). |
| [`attention_ann.md`](attention_ann.md) | The same notebook as rendered Markdown, for reading without a kernel. |

## Running the notebook

It needs only `numpy` and reads no local files, so it runs as-is in Google
Colab with no setup. Locally:

```bash
pip install numpy notebook
jupyter notebook attention_ann.ipynb
```

## Reuse

These materials are free to use, share and adapt, including commercially,
with attribution to TechAarvam. See the notice at the foot of each file.

---

© TechAarvam · <https://www.techaarvam.com>
