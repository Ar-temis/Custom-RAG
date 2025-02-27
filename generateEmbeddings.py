from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("msmarco-distilbert-dot-v5")
newFile = open("new FAQ.txt", 'w')

def extractPairs(doc):
    pairs = []
    currentQ = None
    currentA = []

    file = open(doc)

    for line in file:
        text = line.strip()
        if text.endswith(':'):
            continue
        if text.endswith('?'):
            if currentQ:
                pairs.append(f"{currentQ} {' '.join(currentA)}")
                newFile.write(currentA)
            currentQ = text  # reset the current question and answers
            currentA = []
        elif text.startswith(('●', '○')):
            currentA.append(text.lstrip("●○").strip())
    if currentQ:
        pairs.append(f"{currentQ} {' '.join(currentA)}")

    return pairs


embeddings = model.encode(extractPairs("FAQ.txt"), normalize_embeddings=False)

np.save("FAQ.npy", embeddings)


myQuery = model.encode("How do I get on Dean's list?", normalize_embeddings=False)

hits = util.semantic_search(myQuery,embeddings,top_k=3, score_function=util.dot_score)

for i in hits[0]:
    print(i['corpus_id'])
