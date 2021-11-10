import CodemixedNLP as csnlp

DATASET_FOLDER = "./datasets"

dataset_folder = f"{DATASET_FOLDER}/lince_lid/Hinglish"

pretrained_name_or_path = "bert-base-multilingual-cased"
#pretrained_name_or_path = "xlm-roberta-base"

ckpt_path = "/usr2/home/surajt/CS/CodemixedNLP/datasets/lince_lid/Hinglish/checkpoints/mBERT"

csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "seq_tagging"},
        target_label_fields="langids",
        mode="test",
        eval_ckpt_path=ckpt_path)
