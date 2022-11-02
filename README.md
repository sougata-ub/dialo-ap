# Dialo-AP: A Dependency Parsing Based Argument Parser for Dialogues
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This is the implementation of the paper:

## [**Dialo-AP: A Dependency Parsing Based Argument Parser for Dialogues**](https://aclanthology.org/2022.coling-1.74/)
[**Sougata Saha**](https://www.linkedin.com/in/sougata-saha-8964149a/), [**Souvik Das**](https://www.linkedin.com/in/souvikdas23/), [**Rohini Srihari**](https://www.acsu.buffalo.edu/~rohini/) 

The 29th International Conference On Computational Linguistics (COLING 2022, Gyeongju, Republic of Korea)

## Abstract
While neural approaches to argument mining (AM) have advanced considerably, most of the recent work has been limited to parsing monologues. With an urgent interest in the use of conversational agents for broader societal applications, there is a need to advance the state-of-the-art in argument parsers for dialogues. This enables progress towards more purposeful conversations involving persuasion, debate and deliberation. This paper discusses Dialo-AP, an end-to-end argument parser that constructs argument graphs from dialogues. We formulate AM as dependency parsing of elementary and argumentative discourse units; the system is trained using extensive pre-training and curriculum learning comprising nine diverse corpora.  Dialo-AP is capable of generating argument graphs from dialogues by performing all sub-tasks of AM. Compared to existing state-of-the-art baselines, Dialo-AP achieves significant improvements across all tasks, which is further validated through rigorous human evaluation.

### Datasets Used for Training:
1. [IMHO Corpus](https://aclanthology.org/N19-1054/)
2. [QR Corpus](https://aclanthology.org/D19-1291/)
3. [args.me](https://link.springer.com/chapter/10.1007/978-3-030-30179-8_4)
4. [Feedback Prize Dataset](https://www.kaggle.com/competitions/feedback-prize-2021/overview)
5. [Argumentative Microtext Corpora](https://github.com/peldszus/arg-microtexts)
6. [Consumer Debt Collection Practices](https://aclanthology.org/L18-1257/)
7. [Web Discourse Corpora](https://direct.mit.edu/coli/article/43/1/125/1561/Argumentation-Mining-in-User-Generated-Web)
8. [Persuasive Essays Corpora](https://aclanthology.org/J17-3005/)
9. [Change My View Corpora/Ampersand](https://aclanthology.org/D19-1291/)
10. Combined and formatted dataset used in the experiments:
    1.  For training Dialo-AP: https://dialo-ap-files.s3.amazonaws.com/all_data_combined_curriculum_v4.5.pkl
    2.  For training Ampersand: https://dialo-ap-files.s3.amazonaws.com/imho_formatted_v1.pkl
11. Formatted Change My View dataset used for human evaluation:

### Models:
1. Dialo-AP: https://dialo-ap-files.s3.amazonaws.com/dialo_edu_ap_parser_rerun_0_c_target_dataset.pt
2. Ampersand recreated BERT trained for Component Classification: https://dialo-ap-files.s3.amazonaws.com/bert-base-uncased_imho_component_model.pt
3. Ampersand recreated BERT trained for Inter-Relationship Prediction: https://dialo-ap-files.s3.amazonaws.com/bert-base-uncased_iqr_inter_model.pt
4. Ampersand recreated BERT trained for Intra-Relationship Prediction: https://dialo-ap-files.s3.amazonaws.com/bert-base-uncased_imho_intra_model.pt

### Training and Inference
You can train and evaluate all experiments using the `runner.sh` script. Example: `nohup bash runner.sh 1 12 > log.txt 2>&1 &` runs experiment numbers 1 to 12 sequentially. All the different configurations for the experiments can be found in the `config.json` file.
In order to experiment with different parameters, you can directly execute the `run_training.py` script. Sample command below:

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 9999 ./run_training.py --batch_size 16 --num_epochs 15 --learning_rate 0.00002 --base_transformer "roberta-base"
```
Prior to training, please download the formatted training dataset into a folder named `./data/`.

### Brief description of the files:
1. config.json: contains configuration details of all experiments.
2. runner.sh: parameterized script used to sequentially run all experiments. Sample command: "nohup bash runner.sh 28 34 > log.txt 2>&1 &"
3. models.py: defines all models
4. parser.py: wrapper class which is used for inference
5. trainer.py: contains all training related code
6. run_training.py: parses the parameters used for training

## Citation
If you are using this library then do cite: 
```bibtex
@inproceedings{saha-etal-2022-dialo,
    title = "Dialo-{AP}: A Dependency Parsing Based Argument Parser for Dialogues",
    author = "Saha, Sougata  and
      Das, Souvik  and
      Srihari, Rohini K.",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.74",
    pages = "887--901"
}
```
