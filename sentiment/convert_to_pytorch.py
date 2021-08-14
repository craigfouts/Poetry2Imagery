from transformers import BertTokenizer, BertForSequenceClassification


if __name__ == "__main__":

    model = BertForSequenceClassification.from_pretrained("output/", from_tf=True)

    model.save_pretrained("pytorch_output")

