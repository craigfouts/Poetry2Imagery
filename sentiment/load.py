from datasets import load_dataset




if __name__ == "__main__":
    dataset = load_dataset('poem_sentiment')
    
    dataset['train'].to_csv('train.csv', index=False, columns=["verse_text", "label"])
    dataset['validation'].to_csv('dev.csv', index=False, columns=["verse_text", "label"])
    dataset['test'].to_csv('test.csv', index=False, columns=["verse_text", "label"])

    
    

