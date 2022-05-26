# An enhanced Conv-TasNet model for speech separation using a speaker distance-based loss function


J.A. Arango-Sánchez, and J.D. Arias-Londoño


# Datasets

1. [CallFriend-Caribbean-Spanish-Split](https://auxiliar.s3.us-west-1.amazonaws.com/CallFriend-Caribbean-Spanish-Split.zip)
2. [CallFriend-Spanish-Split](https://auxiliar.s3.us-west-1.amazonaws.com/CallFriend-Spanish-Split.zip)
3. [CallHome-Spanish-Corpus-Split](https://auxiliar.s3.us-west-1.amazonaws.com/CallHome-Spanish-Corpus-Split.zip)

# Train

1. python setup.py
2. python train_model.py --experiment_name "<>" --tags "<>" --save_best_model "<>" --default_root_dir "<>" --epochs <> --num_workers <> --root_csv "data/csv/local/" --batch_size <> --weight_CS <> --num_layers <>


# Test

1. [Download checkpoints](https://drive.google.com/file/d/1cCEWfunIkVQQU7f3cXWIp8m6jicMyFhK/view?usp=sharing) 
2. python test_model.py --experiment_name "<>" --tags "<>" --root_csv "<>" --save_best_model "MODEL_PATH"
