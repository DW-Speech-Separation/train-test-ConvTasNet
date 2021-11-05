# train-test-ConvTasNet

### Experiment name
<Train/Test> <Dataset(CallFriend/All)> <tipo de loss> <tipo de embedding>


# Train

1. python setup.py
2. cd src/models/
3. !python train_model.py --experiment_name "Entrenamiento modelo original con dataset CallFriend con loss original SI-SDR, sin embeddign" --tags "Modelo original SIN embedding" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/" --default_root_dir "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/" --epochs 2 --num_workers 2


# ALL

!python train_model.py --experiment_name "Entrenamiento modelo original \
con dataset All con loss original SI-SDR + loss_similarity dada por el coseno similarity usando Wav2Vect Speech Embedding mas peso 5,  " --tags "Modelo con WaV2Vec embedding" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_5/" --default_root_dir "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_5" --epochs 100 --num_workers 2 --weight_CS 5 --batch_size 6 --train_csv "ALL_mixture_train_mix_clean_spanish.csv" --valid_csv "ALL_mixture_val_mix_clean_spanish.csv" --test_csv "ALL_mixture_test_mix_clean_spanish.csv"



!python train_model.py --experiment_name "Entrenamiento modelo original \
con dataset All con loss original SI-SDR + loss_similarity dada por el coseno similarity usando Wav2Vect Speech Embedding mas peso 10,  " --tags "Modelo con WaV2Vec embedding" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_10/" --default_root_dir "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_10" --epochs 100 --num_workers 2 --weight_CS 10 --batch_size 6 --train_csv "ALL_mixture_train_mix_clean_spanish.csv" --valid_csv "ALL_mixture_val_mix_clean_spanish.csv" --test_csv "ALL_mixture_test_mix_clean_spanish.csv"



!python train_model.py --experiment_name "Entrenamiento modelo original \
con dataset All con loss original SI-SDR + loss_similarity dada por el coseno similarity usando Wav2Vect Speech Embedding mas peso 20,  " --tags "Modelo con WaV2Vec embedding" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_20/" --default_root_dir "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_ALL_Dataset_Sum_Loss/ALL_weigth_20" --epochs 100 --num_workers 2 --weight_CS 20 --batch_size 6 --train_csv "ALL_mixture_train_mix_clean_spanish.csv" --valid_csv "ALL_mixture_val_mix_clean_spanish.csv" --test_csv "ALL_mixture_test_mix_clean_spanish.csv"




## Test Model

!python test_model.py --experiment_name "Test modelo original con dataset CallFriend con loss original SI-SDR + loss_similarity dada por el coseno similarity usando Wav2Vect Speech Embedding mas peso 5" --tags "Test con WaV2Vec embedding peso 5" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/callFriend_weigth_5/"