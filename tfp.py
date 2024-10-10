import os
import json
import sys
import numpy as np
import torch
from transformers import BertTokenizer, AutoModel

@torch.no_grad()
def bert_embedding(texts, batch_size=100):
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    model = AutoModel.from_pretrained('google-bert/bert-base-uncased').cuda()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        all_embeddings.append(cls_embeddings.cpu())
        print(f"Processed {min(i + batch_size, len(texts))} / {len(texts)} texts")

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings.numpy()

def tfp_algorithm(embeddings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)

    n = embeddings.size(0)
    visited = torch.zeros(n, dtype=torch.bool, device=device)
    path = [0]
    visited[0] = True

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # Get all pairwise distances excluding zeros (self-distances)
    all_distances = dist_matrix[dist_matrix > 0]
    threshold = torch.quantile(all_distances, 0.02).item()  # 2nd percentile as threshold

    invalid_attempts = 0

    for _ in range(1, n):
        print('Current path length:', len(path))
        last = path[-1]
        distances = dist_matrix[last].clone()
        distances[visited] = float('inf')  # Exclude visited nodes
        next_index = torch.argmin(distances).item()

        max_attempts = 10000  # Maximum number of attempts to find a valid next node
        attempts = 0

        while True:
            valid = True
            # Ensure the next node is not too close to the last up to 4 nodes
            for back in range(1, min(5, len(path)+1)):
                prev_index = path[-back]
                if dist_matrix[prev_index, next_index] < threshold:
                    valid = False
                    break

            if valid:
                print('Number of invalid attempts:', invalid_attempts)
                break
            else:
                invalid_attempts += 1
                distances[next_index] = float('inf')
                next_index = torch.argmin(distances).item()

            attempts += 1
            if attempts >= max_attempts:
                print(f"Exiting loop after {max_attempts} attempts.")
                break

        path.append(next_index)
        visited[next_index] = True

    print(f"Total invalid attempts: {invalid_attempts}")
    return path

def main(input_file, output_file):
    with open(input_file, "r") as fp:
        data = json.load(fp)

    instruction_list = [d["instruction"] for d in data]
    print('Processing instructions...')

    embedding_file = "bert_embedding.npy"
    if os.path.exists(embedding_file):
        text_embedding = np.load(embedding_file)
        print("Loaded embeddings from file.")
    else:
        text_embedding = bert_embedding(instruction_list)
        np.save(embedding_file, text_embedding)
        print("Computed and saved embeddings.")

    res = tfp_algorithm(text_embedding)

    reordered_data = [data[index] for index in res]
    with open(output_file, "w") as fp:
        json.dump(reordered_data, fp, indent=2, ensure_ascii=False)

    print('Processing complete. Output saved to', output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_file output_file")
        sys.exit(1)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        if not os.path.exists(input_file):
            print(f"Error: input file '{input_file}' does not exist.")
            sys.exit(1)
        main(input_file, output_file)
