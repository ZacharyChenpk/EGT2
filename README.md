# EGT2: Entailment Graph Learning with Textual Entailment and Soft Transitivity

This repository contains the source code of the ACL 2022 long paper "Entailment Graph Learning with Textual Entailment and Soft Transitivity". [paper](http://arxiv.org/abs/2204.03286)

## How to run

1. Download the evaluation [repository](https://github.com/mjhosseini/entgraph_eval) and the test dataset for later usage:

```
git clone https://github.com/mjhosseini/entgraph_eval.git
wget https://dl.dropboxusercontent.com/s/j7sgqhp8a27qgcf/gfiles.zip
unzip gfiles.zip
rm gfiles.zip
ls ../gfiles/ent/
```

2. Follow the Hosseini ACL2019 [repository](https://github.com/mjhosseini/linkpred_entgraph) to get local triples and base graphs for later usage. Suppose that you have a graph filefolder now:

```
ls ../gfiles/typedEntGrDir_NS_all_AUG_MC/
```

3. Run the corpus generator with sentence generator $S$ to transfer the dataset into natural sentences:

```
python corpus_gen.py
```

4. Run the DeBERTa trainer to get the pair-wise local scorer $LM(entail|p,q)$:

```
python deberta_finetune.py --n_epoch 300
```

or directly use the trained [model](https://drive.google.com/file/d/19BzTyyzsRUqa_HIHhgwMMEQbSgGVmPHC/view?usp=sharing):

```
mkdir deberta_tars
mv deberta0.8_12_1e-05_1_best.pth.tar deberta_tars/
```

5. Change the relative path in gf_bertnli_modifier.py Line 334, and run the local graph generator:

```
python gf_bertnli_modifier.py
```

6. Run the global graph generator:

```
python new_global_factory_split.py --use_cuda 1 --use_cross 0 --use_reso 0 --lr 0.05 --lambda1 0 --epsilon1 0.2 --lambda2 2 --lambda_trans 1 --epsilon_trans 0.98 --trans_method 3 --featIdx 0 --gpath typedEntGrDir_bertnli_modified_aug_tuned_r0.8 --writepath typedEntGrDir_EGT2
```

7. Evaluation by the [repository](https://github.com/mjhosseini/entgraph_eval).

```
cd ~/entgraph_eval/evaluation/
python eval.py --gpath typedEntGrDir_EGT2 --test --sim_suffix _sim.txt --method EGT2 --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --exactType --backupAvg --write
```

8. In some cases, as those low-confidence pairs might be set to zero, and therefore the lowest precision will higher than 0.5, leading to the incorrectly evaluation and lower metrics. So run the curve filler to get the right result.

```
ls ~/gfiles/EGT2/
python eval_curvefill.py
```