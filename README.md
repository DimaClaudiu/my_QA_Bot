# QA Pipeline
Natural language search through a large knowledge base(KB).

![](https://paper-attachments.dropbox.com/s_60F722FD326451DB6F40490B2F3931616DFAE97900230B4966BBE26C41DE7EDA_1600975194750_schem-small.png)

## Components

So far, the pipeline is mainly split in:

1. **Categorical contexts**: The knowledge base is assumed to be clustered in a finite number of classes. 
    For example, chats groups are categorized per teams and actions:
    - ML Team
    - VPN Team
    - Lunch Food Orders
    - Music
    - ...
    Each of those represents a subdomain of our KB.
    For now, they are stored in independent text files. But any database solution can easily replace this component.
----------
2. **Classifier**: A model that helps narrow the searching field by offering the most likely sub-domain of any given question.
    Examples:
    Q: *What's the snakes' favorite song?*
    A: *MUSIC*
    
    Q: *Where can I get the APK for Hubgets?*
    A: *ANDROID APP*


    For now, I'm using a base RoBERTa model, fine-tunned for multiclass-classification.
    
    For training, I used [News Category Dataset v2](https://www.kaggle.com/rmisra/news-category-dataset), taking just the first 10 most popular categories. The dataset fits nicely because the data is quite sparse and out of context (similar to text chats, where conversations are short and incomplete) and the classes are alose quite imbalanced (also similar to real-world data).
    The data was cleaned before training, since common words would give an unfair advantage to larger(in size of text) classes.
    This component has to be retrained for each new KB, and also updated periodically.
----------
3. **Ranker**: A second component that helps narrow down the search field.
    Given a body of text and a question, for example, all the messages in a group, this components groups the messages into max-length spans of text (by date, or by hours, since context should be roughly preserved on a basis of time) and assigns a probability for each span, of how likely the answer resides there. 


    In abstract words, the component gives the best candidate paragraphs for a given question, from a given corpus of text.
    
    For now, this component uses the Tfidf algorithm to find paragraphs similar to the question. This is a problem since it relies on matching words between the contexts and questions.
    
    A different approach I'm currently experimenting with is using a model for Dense Passage Retrieval, as described [here](https://arxiv.org/abs/2004.04906). This will use dense embeddings of the contexts and question, and score the similarity using the dot product of the embeddings, instead of cosine similarity.
    
    This will allow finding candidates based on conceptual overlap, not just semantic similarities. For example:
    C: *...I love bananas because they are yellow...*
    Q: *When did we talk about fruits?*
    
    This model should score that context pretty highly since a banana is indeed a fruit.
    BM25 could be also used for a fast and reliable ranker.
----------
4. **Reader**: The last component is an extractive, closed domain, Question Answer Model.


    It receives a context and a question and extracts the answer.
    This model is also a fine-tunned RoBERTa-base for Question-Answering.
    
    Initially, I tried a pre-trained BERT model, trained on SQuAD 1.1, the model was great at extracting the answers when those existed, but when the answer wasn't there, it was really confident in giving the wrong ones :(
    
    
    
    SQUaD 2.0 addresses these issues by having impossible answers, pairs of , where the questions simply can't be answered, and the model should output blank ''. 
    So I trained a fine-tuned RoBERTa for this downstream task on squad2.
    
    The model performs great, but it's quite slow. That's why the previous components try to filter out as much junk as possible for this one.
----------
## Stats & Metrics:
1. **Classifier**:
    - POLITICS: 32738
    - WELLNESS: 17826
    - ENTERTAINMENT: 16057
    - TRAVEL: 9886
    - STYLE & BEAUTY: 9648
    - PARENTING: 8676
    - HEALTHY LIVING: 6693
    - QUEER VOICES: 6313
    - FOOD & DRINK: 6225
    - BUSINESS: 5936
    
| Loss (CategoricalCrossentropy) | Metric (CategoricalAccuracy) |
| ------------------------------ | ---------------------------- |
| 0.6720663905143738             | 0.7869972586631775           |

     train/test split: 0.2
     epochs: 6
     batch_size: 24


2. **Ranker**: N/A



3. **Reader**:


| Loss (CategoricalCrossentropy) | Metric (CategoricalAccuracy) |
| ------------------------------ | ---------------------------- |
| 0.611580415368080158           | 83.6129032258064448          |

     train/test split: 0.2
     epoch 3
     batch_size: 8
    
----------
## References:

**APIs & Implementations**
Transformers library: https://github.com/huggingface/transformers

API for fast NLP prototyping: https://github.com/ThilinaRajapakse/simpletransformers



DPR implementation: https://github.com/facebookresearch/DPR
**Papers**
Attention is all you need paper for BERT: https://arxiv.org/abs/1706.03762

RoBERTa paper: https://arxiv.org/abs/1907.11692
DPR paper: https://arxiv.org/abs/2004.04906facebook 
**Articles**
BERT for QA: https://medium.com/saarthi-ai/build-a-smart-question-answering-system-with-fine-tuned-bert-b586e4cfa5f5

Architecture comparison: https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8
**Datasets**
SQUaD and SQuAD2: https://rajpurkar.github.io/SQuAD-explorer/

News dataset: https://www.kaggle.com/rmisra/news-category-dataset

