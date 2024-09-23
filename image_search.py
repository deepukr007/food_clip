from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel , CLIPTokenizer
import warnings
import numpy as np 
import os
import matplotlib.pyplot as plt
np.random.seed(42)
import torch

warnings.filterwarnings("ignore")


class ImageSearch:
    def __init__(self, dataset):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.imageprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dataset = load_dataset(dataset)
    

    def embed_images(self, mode="random" , batch_size=16):
        if os.path.exists("image_embeddings.npy"):
            self.image_embeddings = np.load("image_embeddings.npy")
            return self.image_embeddings
        else:
            
            self.embeddings= []

            for i in range(0, len(self.dataset["train"]), batch_size):
                idx = range(i, i+batch_size)
                batch = [self.dataset["train"][i]["image"] for i in idx if i < len(self.dataset["train"])]
                inputs = self.imageprocessor(text=None , images=batch, return_tensors="pt", padding=True).to('cuda')
                outputs = self.model.get_image_features(**inputs)
                self.embeddings.append(outputs.detach().cpu().numpy())
            self.image_embeddings = np.concatenate(self.embeddings)
            np.save("image_embeddings.npy", self.image_embeddings)
    
    def embed_text(self, text):
        input = self.tokenizer(text, return_tensors="pt", padding=True).to('cuda')
        outputs = self.model.get_text_features(**input)
        self.text_embedding = outputs.detach().cpu().numpy()
        return self.text_embedding

    def search(self, text, n_results=5):
        text_embedding = self.embed_text(text)
        scores = (self.image_embeddings @ text_embedding.T).squeeze(1)
        top_idx = scores.argsort()[::-1][:n_results].tolist()
        return [self.dataset["train"][i] for i in top_idx]


if __name__ == "__main__":
    isearch = ImageSearch(dataset="EduardoPacheco/FoodSeg103")
    isearch.embed_images()

    results = isearch.search("Pasta bolognese")
    top_result = results[0]
    plt.imshow(top_result["image"])
    plt.show()
