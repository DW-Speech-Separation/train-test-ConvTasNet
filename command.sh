python train_model.py --experiment_name "Entrenamiento modelo original \
con dataset CallFriend con loss original SI-SDR + loss_similarity dada por el coseno similarity usando Wav2Vect Speech Embedding mas peso 10 y 1 transformer,  " --tags "Modelo con WaV2Vec embedding" --save_best_model "models/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/callFriend_weigth_10/1_transformer" --default_root_dir "models/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/callFriend_weigth_10/1_transformer" --epochs 100 --num_workers 4 --weight_CS 10 --batch_size 6 --num_layers 1 --root_csv "data/csv/local/"