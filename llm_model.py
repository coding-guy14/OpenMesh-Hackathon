from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#region Setup
"""
GPT2Model is a class that represents a sentiment prediction model based on the GPT-2 language model.
Attributes:
    tokenizer (GPT2Tokenizer): The tokenizer used for encoding text.
    model (GPT2LMHeadModel): The GPT-2 language model used for sentiment prediction.
Methods:
    predict_sentiment(prompt: str) -> str:
        Predicts the sentiment of a given prompt and returns the sentiment as a string.
    extract_sentiment(text: str) -> str:
        Extracts the sentiment from a given text and returns the sentiment as a string.
"""
class GPT2Model:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")


    """
    Predicts the sentiment of a given prompt and returns the sentiment as a string.
    Args:
        prompt (str): The prompt for which sentiment needs to be predicted.
    Returns:
        str: The predicted sentiment as a string.
    """
    def predict_sentiment(self, prompt: str) -> str:
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long) # create an attention mask
        pad_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(inputs, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=50, num_return_sequences=1)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.extract_sentiment(text)


    """
    Extracts the sentiment from a given text and returns the sentiment as a string.
    Args:
        text (str): The text from which sentiment needs to be extracted.
    Returns:
        str: The extracted sentiment as a string.
    """
    @staticmethod
    def extract_sentiment(text: str) -> str:
       
        if "positive" in text.lower():
            return "Positive"
        elif "negative" in text.lower():
            return "Negative"
        else:
            return "Neutral"
#endregion