# reddit-argument-parser
config.json: contains configuration details of all experiments.
runner.sh: parameterized script used to sequentially run all experiments. Sample command: "nohup bash runner.sh 28 34 > log.txt 2>&1 &"
models.py: defines all models
parser.py: wrapper class which is used for inference
trainer.py: contains all training related code
run_training.py: parses the parameters used for training