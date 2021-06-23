## Demo huggingface with wandb Tables

* you may need a GPU

* install dependencies from `requirements.txt`

* you can explore `notebook.ipynb` to understand how to access a `Dataset`, most common class for input in huggingface

* run following command

  `python train.py --model_name_or_path facebook/wav2vec2-large-xlsr-53 --dataset_config_name tr --train_split_name train --output_dir ./model --overwrite_output_dir --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 8 --fp16 --freeze_feature_extractor --group_by_length --gradient_checkpointing --do_train --save_total_limit 1 --logging_steps 1 --warmup_steps 500 --activation_dropout=0.030711785983032657 --attention_dropout=0.12333330186818241 --feat_proj_dropout=0.021573944881012816 --gradient_accumulation_steps=4 --hidden_dropout=0.02844812528063585 --layerdrop=0.012990322756361196 --learning_rate=0.00011222484369340102 --mask_time_prob=0.09237120620890814 --do_eval --evaluation_strategy epoch --load_best_model_at_end --metric_for_best_model wer --greater_is_better False --max_samples 100`

* for longer training, just remove `--max_samples 100`

* first run will be slow due to data preprocessing but it will be cached for next runs

* after using this repo, you may want to remove the data from `~/.cache/huggingface/`

* at the moment, we mainly need to make the notebook work. Then we can use it in the script (I added `TODO` for the places that will need `ValidationDataLogger`)

## Random comments

* it would be nice to simply call `from wandb import ValidationDataLogger`
* should it just be called `DataLogger` (I may want to use it with test data or even train data)
* should `validation_row_processor` be named `input_processor`
* same with `prediction_processor`
* not sure about `input_col_name`, can I have several columns as inputs (see example notebook)?
* I have an evaluation dataset and a test dataset, I'm thinking of using 2 `ValidationDataLogger`
* In this case, they are actually the same dataset (but processed differently for training purposes) so I could potentially use the same table (but would like to differentiate them with 2 output columns)
