# Text sentiment classifier

Trained on Huggingface datasets 'poem_sentiment' dataset.

## Dependencies

tensorflow==2.3.3
transformers==4.9.1
datasets==1.11.0

## training process

run `python load.py` to create `train.csv`, `dev.csv`, and `test.csv`.

using `run_text_classification.py` from [huggingface/transformers examples](https://github.com/huggingface/transformers/blob/4a872caef4e70595202c64687a074f99772d8e92/examples/tensorflow/text-classification/run_text_classification.py)

```bash
python run_text_classification.py --model_name_or_path bert-large-cased --train_file train.csv --do_train --do_eval --do_predict --validation_file dev.csv --test_file test.csv --output_dir output --logging_dir log --num_train_epochs 4 --overwrite_output_dir --seed 25 --learning_rate 0.00001 --lr_scheduler linear
```

### Model considerations

The model can deal with text up to 128 tokens long.


### sentiment-cli tool

The `sentiment-cli` tool uses an [ONNX](https://github.com/onnx/onnx/) model that depends on the [onnxruntime](https://github.com/microsoft/onnxruntime/).

### examples of sentiment-cli tool

```
sentiment model/model.onnx "me honied paths forsake;" -t vocab.txt 
me honied paths forsake; : Neutral 0.62159
```

```
sentiment model/model.onnx "when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals," -t vocab.txt 
when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals, : Mixed 0.44403
```

```
sentiment model/model.onnx "and that is why, the lonesome day," -t vocab.txt 
and that is why, the lonesome day, : Negative 0.94373
```
