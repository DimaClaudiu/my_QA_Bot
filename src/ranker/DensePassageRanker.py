from ranker.ranker import Ranker

from transformers import DPRReader, DPRReaderTokenizer
import torch


class DensePassageRanker(Ranker):

    def __init__(self):
        self.tokenizer = DPRReaderTokenizer.from_pretrained(
            'facebook/dpr-reader-single-nq-base')
        self.model = DPRReader.from_pretrained(
            'facebook/dpr-reader-single-nq-base', return_dict=True)

    def rank(self, question, contexts, top_k):
        encoded_inputs = self.tokenizer(
            questions=[question],
            titles=[""],
            texts=contexts,
            return_tensors='pt'
        )
        input_ids = encoded_inputs["input_ids"].tolist()[0]

        outputs = self.model(**encoded_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        relevance_logits = outputs.relevance_logits

        answer_start = torch.argmax(
            start_logits
        )

        answer_end = torch.argmax(end_logits) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer
