from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel , CLIPTokenizer
import warnings
import numpy as np 
import os
import matplotlib.pyplot as plt
np.random.seed(42)

warnings.filterwarnings("ignore")

class ImageSearch:
    def __init__(self, dataset):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.imageprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dataset = load_dataset(dataset)
    

    def embed_images(self, mode="random", n_images=100 , batch_size=32):
        self.random_idx = np.random.randint(0, len(self.dataset["train"]), n_images).tolist()
        if os.path.exists("image_embeddings.npy"):
            self.image_embeddings = np.load("image_embeddings.npy")
            return self.image_embeddings
        else:
            
            images = [self.dataset["train"][i]["image"] for i in self.random_idx]

            self.embeddings= []

            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                inputs = self.imageprocessor(text=None , images=batch, return_tensors="pt", padding=True)
                outputs = self.model.get_image_features(**inputs)
                self.embeddings.append(outputs.detach().cpu().numpy())
            self.image_embeddings = np.concatenate(self.embeddings)
            np.save("image_embeddings.npy", self.image_embeddings)
    
    def embed_text(self, text):
        input = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model.get_text_features(**input)
        self.text_embedding = outputs.detach().cpu().numpy()
        return self.text_embedding

    def search(self, text, n_results=5):
        text_embedding = self.embed_text(text)
        scores = (self.image_embeddings @ text_embedding.T).squeeze(1)
        top_idx = scores.argsort()[::-1][:n_results]
        dataset_ids = [ self.random_idx[i] for i in top_idx]
        return [self.dataset["train"][i] for i in dataset_ids]


if __name__ == "__main__":
    isearch = ImageSearch(dataset="EduardoPacheco/FoodSeg103")
    isearch.embed_images()

    results = isearch.search("Pasta bolognese")
    top_result = results[0]
    plt.imshow(top_result["image"])
    plt.show()
