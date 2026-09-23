# AGAR Colony Vision

Two experimental directions built on the [AGAR dataset](https://agar.neurosys.com/) —
a large annotated collection of Petri-dish photos of 5 microbial species
(`S.aureus`, `B.subtilis`, `P.aeruginosa`, `C.albicans`, `E.coli`).

## [`detection/`](detection) — direct colony detection

A YOLOv8/YOLOv10 object detector trained to localize and classify colonies on Petri-dish
images, plus a live webcam detection app.

**Result:** P=0.930, R=0.870, mAP50=0.923 (YOLOv8n, 20 epochs, 640px).

## [`synthetic-data-generation/`](synthetic-data-generation) — style-transfer data augmentation

An adaptation of NeuroSYS's synthetic AGAR image generator (colony extraction + composition +
neural style transfer), explored as a way to grow the detection training set beyond the raw
AGAR images. Runs end-to-end after fixing several library-version compatibility issues the
original 2021 code had accumulated. A full retrain-and-compare against the detection baseline
is still open — see that folder's README for the compute cost involved and why it's not done yet.

## Data & weights

Neither the raw AGAR images nor trained model weights are checked into this repo — AGAR has
its own license/citation terms (see [agar.neurosys.com](https://agar.neurosys.com/)), and the
weights are easy to regenerate from `detection/train.py`. A handful of representative result
plots and generated samples are included for reference.

## Citations

If you use AGAR or the style-transfer generation approach, cite the original work:

> S. Majchrowska et al., *AGAR a microbial colony dataset for deep learning detection*,
> arXiv:2108.01234 (2021).

> J. Pawłowski, S. Majchrowska, & T. Golan, *Generation of microbial colonies dataset with
> deep learning style transfer*, Scientific Reports 12, 5212 (2022).

## License

Code in this repo is MIT-licensed (see [LICENSE](LICENSE)). This does not cover the AGAR
dataset itself, which has its own terms.
