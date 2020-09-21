from ranker.ranker import Ranker

from transformers.modeling_dpr import DPRContextEncoder, DPRQuestionEncoder
from transformers.tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer


class DensePassageRanker(Ranker):
    def __init__(self, question_encoder_path="facebook/dpr-question_encoder-single-nq-base", context_encoder_path="facebook/dpr-ctx_encoder-single-nq-base"):

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder_path)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder_path).to(self.device)

        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            context_encoder_path)
        self.context_encoder = DPRContextEncoder.from_pretrained(
            context_encoder_path).to(self.device)
