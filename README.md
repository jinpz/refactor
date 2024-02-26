# REFACTOR: Learning to Extract Theorems from Proofs

This is the official repo for

[**REFACTOR: Learning to Extract Theorems from Proofs**](https://openreview.net/forum?id=fgKjiVrm6u) (ICLR 2024)

by Jin Peng Zhou*, Yuhuai Wu*, Qiyang Li, and Roger Grosse \
*Equal Contribution

## Abstract
Human mathematicians are often good at recognizing modular and reusable theorems that make complex mathematical results within reach. In this paper, we propose a novel method called theoREm-from-prooF extrACTOR (REFACTOR) for training neural networks to mimic this ability in formal mathematical theorem proving. We show on a set of unseen proofs, REFACTOR is able to extract 19.6% of the theorems that humans would use to write the proofs. When applying the model to the existing Metamath library, REFACTOR extracted 16 new theorems. With newly extracted theorems, we show that the existing proofs in the MetaMath database can be refactored. The new theorems are used very frequently after refactoring, with an average usage of 733.5 times, and help shorten the proof lengths. Lastly, we demonstrate that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.

## Theorem Expansion
To expand theorems in `set.mm` or `propositional.mm`:
```
python theorem_expansion.py
```
Various data splitting and capping options can be specified in the `theorem_expansion.py` file.

## Theorem Extraction Model Training
```
python train.py
```
Detailed hyperparameter options can be found in the paper and `train.py` file.

## Theorem Verification

To extract and verify new theorems predicted by REFACTOR model
```
python theorem_verification.py
```
The main extraction and verification code is in the function `analyze_predictions`. We also perform standardization and deduplication of the extracted theorems.
The code will output an augmented `set.mm` file with the new theorems and their proofs.

## Theorem Refactoring
Using the augmented `set.mm` file we can refactor the existing proofs in the Metamath database.
```
python theorem_refactor.py
```
One example of refactored augmented theorem file is in `refactor_dataset/augmented_set_refactored.mm`.

## Theorem Proving with Refactored Theorems
We use [MetaGen's](https://arxiv.org/abs/2002.07019) implementation of the [Holophrasm](https://arxiv.org/abs/1608.02644) and train a prover on our refactored theorems. For more details, please see this [repo](https://github.com/princeton-vl/MetaGen). We provide the proved theorems in `holophrasm_proved_theorems/`.

## Citation
If you find this code useful in your research, please consider citing:
```
@inproceedings{zhou2023refactor,
  title={REFACTOR: Learning to Extract Theorems from Proofs},
  author={Zhou, Jin Peng and Wu, Yuhuai and Li, Qiyang and Grosse, Roger Baker},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
