# Text sentiment classifier

Trained on Huggingface datasets 'poem_sentiment' dataset.

## Dependencies

tensorflow==2.3.3
transformers==4.9.1
datasets==1.11.0

## training process

run `python load.py` to create `train.csv`, `dev.csv`, and `test.csv`.

```bash
python run_text_classification.py --model_name_or_path bert-large-cased --train_file train.csv --do_train --do_eval --do_predict --validation_file dev.csv --test_file test.csv --output_dir output --logging_dir log --num_train_epochs 4 --overwrite_output_dir --seed 25 --learning_rate 0.00001 --lr_scheduler linear
```

