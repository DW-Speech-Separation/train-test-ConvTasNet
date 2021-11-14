
import yaml
import json
import pandas as pd
import os
import numpy as np
import random as random
import torch
from tqdm import tqdm
import soundfile as sf
from asteroid import ConvTasNet
from src.data.CallSpanish_dataset import CallSpanish
from pytorch_lightning.loggers.neptune import NeptuneLogger
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.dsp.normalization import normalize_estimates
from src.config.base_options import BaseOptions
from torch.utils.data import DataLoader
from asteroid.utils import tensors_to_device
from asteroid.metrics import get_metrics
import neptune.new as neptune

class Test:
    def __init__(self,opt):
        self.opt =opt
        self.PATH_CONFIG = opt.path_config
        self.conf = self.open_config()
        self.experiment_name = opt.experiment_name
        self.tags= opt.tags
        self.ROOT_CSV = opt.root_csv
        self.PATH_CSV_TEST = self.ROOT_CSV + opt.test_csv
        self.results = np.zeros(10)
        self.df_test = pd.read_csv(self.PATH_CSV_TEST)
        self.neptune_logger= self.initialize_neptune()
        start = 0
        end = self.df_test.shape[0]
        self.start = start
        self.end = end
        self.exp_dir = opt.save_best_model

    def open_config(self):
        with open(self.PATH_CONFIG) as f:
            conf = yaml.safe_load(f)
            return conf

    def create_test_dataset(self):
        return CallSpanish(
            csv_path=self.PATH_CSV_TEST,
            task="sep_clean",
            sample_rate=8000,
            n_src=2,
            segment=None,
            return_id=True
        )
        

    def initialize_neptune(self):
        
        # Configurarmos el experimento y sus parametros
        experiment_name = self.experiment_name
        params=self.conf
        tags = [self.tags]

        # Definir Logger 
        neptune_logger = neptune.init(
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NjRkMmY2YS04M2EwLTRiMGMtODk1Ny1mMWQxZTA3NGM1NzAifQ==",
            project="josearangos/Tg-speech-separation",name=experiment_name,
            tags = tags)

        return neptune_logger


    def step_test(self,start, results,model,loss_func,COMPUTE_METRICS,eval_save_dir,ex_save_dir,test_set,neptune_status,model_device,pretrained=False):
        series_list = []

        i = start

        torch.no_grad().__enter__()
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, ids = test_set[idx]
            mix, sources = tensors_to_device([mix, sources], device=model_device)
            est_sources = model(mix.unsqueeze(0))
            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()

            utt_metrics = get_metrics(
                        mix_np,
                        sources_np,
                        est_sources_np,
                        sample_rate=self.conf["data"]["sample_rate"],
                        metrics_list=COMPUTE_METRICS,
                    )


            r = 0
            iteration = str(i)
            #self.neptune_logger.experiment.log_metric("Iteración:", i)
            self.neptune_logger['Iteracion'].log(i)

            for metric_name in COMPUTE_METRICS:
                input_metric_name = "input_" + metric_name     
                results[r] = results[r] + utt_metrics[input_metric_name]                   
                # iteration + "_"+
                #self.neptune_logger.experiment.log_metric(input_metric_name, results[r])
                self.neptune_logger[input_metric_name].log(results[r])
                r = r +1
                

            for metric_name in COMPUTE_METRICS:
                results[r] = results[r] + utt_metrics[metric_name]          
                #iteration + "_"+ 
                #self.neptune_logger.experiment.log_metric(metric_name, results[r])
                self.neptune_logger[metric_name].log(results[r])

                r = r +1
            
                
       
            utt_metrics["mix_path"] = test_set.mixture_path
            series_list.append(pd.Series(utt_metrics))


            i = i + 1 

            est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)


            # Save some examples in a folder. Wav files and metrics as text.
            if self.conf["test"]["n_save_examples"] == -1:
                self.conf["test"]["n_save_examples"] = len(test_set)
            
            save_idx = random.sample(range(len(test_set)),self.conf["test"]["n_save_examples"])

            """
            if idx in save_idx:

                example_name = "ex_{}/".format(idx)

                local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
                os.makedirs(local_save_dir, exist_ok=True)
                
                
                sf.write(local_save_dir + "mixture.wav", mix_np, self.conf["data"]["sample_rate"])

                
                
                # Loop over the sources and estimates
                for src_idx, src in enumerate(sources_np):
                    sf.write(local_save_dir + "s{}.wav".format(src_idx), src, self.conf["data"]["sample_rate"])
                for src_idx, est_src in enumerate(est_sources_np_normalized):
                    path_estimation_source = local_save_dir + "s{}_estimate.wav".format(src_idx)

                    if (pretrained):
                        path_estimation_source = local_save_dir + "s{}_pretrained_estimate.wav".format(src_idx)
                                    
                    sf.write(path_estimation_source,
                            est_src,
                            self.conf["data"]["sample_rate"],
                    )
                    if (neptune_status):
                        #self.neptune_logger.experiment.log_artifact(path_estimation_source)
                        nam = path_estimation_source.split(".")[0]
                        self.neptune_logger['examples/'+nam+"/"].upload(path_estimation_source)


                #Send estimation wavs
                mix_path = local_save_dir + "mixture.wav"
                if (neptune_status):
                    #self.neptune_logger.experiment.log_artifact(mix_path)
                    nam = mix_path.split(".")[0]
                    self.neptune_logger['examples/'+nam+"/"].upload(mix_path)

                neptune_status = False
                        
                # Write local metrics to the example folder.
                with open(local_save_dir + "metrics.json", "w") as f:
                    json.dump(utt_metrics, f, indent=0)
        """

        return series_list



    def compute_global_metrics(self,start,end, series_list,COMPUTE_METRICS,eval_save_dir,pretrained=False):
        all_metrics_df = pd.DataFrame(series_list)
        name = "all_metrics.csv"
        final_metrics = "final_metrics.json"
        if (pretrained):
            name= "all_metrics_pretrained_model_start_"+str(start)+"_end_"+str(end)+".csv"
            final_metrics ="final_metrics_pretrained_model_start_"+str(start)+"_end_"+str(end)+".json"
            
        all_metrics_path = os.path.join(eval_save_dir,name )  
        all_metrics_df.to_csv(all_metrics_path)

        #Send All metrics
        #self.neptune_logger.experiment.log_artifact(all_metrics_path)
        nam = all_metrics_path.split(".")[0]
        self.neptune_logger['all_metrics'].upload(all_metrics_path)

        final_results = {}
        for metric_name in COMPUTE_METRICS:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
       

        summary_metrics = os.path.join(eval_save_dir,final_metrics)


        with open(summary_metrics, "w") as f:
                json.dump(final_results, f, indent=0)

        #Send summary metrics
        #self.neptune_logger.experiment.log_artifact(summary_metrics)
        nam = summary_metrics.split(".")[0]
        self.neptune_logger['final_metrics'].upload(summary_metrics)


    def run_test(self,start, end, results, model,test_set,pretrained=True):
        model_device = next(model.parameters()).device
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]

        eval_save_dir = os.path.join(self.exp_dir, "metrics")
        os.mkdir(eval_save_dir)
        ex_save_dir = os.path.join(eval_save_dir, "examples/")
        neptune_status=True

        series_list = self.step_test(start, results, model,loss_func,COMPUTE_METRICS,eval_save_dir,ex_save_dir,test_set,neptune_status,model_device,pretrained)
        self.compute_global_metrics(start,end,series_list,COMPUTE_METRICS,eval_save_dir,pretrained)  
        self.neptune_logger.stop()

    def loading_model(self):
        path_best_model = os.path.join(self.exp_dir, "best_model.pth")
        best_model  = ConvTasNet.from_pretrained(path_best_model)
        best_model = best_model.cuda()
        return best_model

    def run(self):
        #Create dataset
        print("=="*30)
        print("Create dataset...")
        test_set = self.create_test_dataset()

        print("=="*30)
        print("Loading model...")    
        #Load model from checkpoints
        best_model = self.loading_model()

        #Run test, use neptune to report metrics and monitoring experiment
        #Guardar las metricas
        print("=="*30)
        print("Run test...")
        self.run_test(self.start, self.end, self.results, best_model,test_set,pretrained=True)


if __name__ == '__main__':
    # Read paramatres of command line
    opt = BaseOptions().parse()
    Test(opt).run()

