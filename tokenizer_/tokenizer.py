import os


class Series_Tokenizer:

    def __init__(self, mode, mecab_dic_path="C:\mecab\mecab-ko-dic"):
        self.mode = mode
        if mode == "mecab":
            if os.path.isdir(mecab_dic_path):
                from konlpy.tag import Mecab
                m = Mecab(mecab_dic_path)
                self.tokenizer = m.morphs
            else:
                print("Not exist 'mecab_dic_path'")

        elif mode == "kobert":
            """
            SAME TO:
            from pytorch_transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            tokens = tokenizer.tokenize(sentence)
            indexes = tokenizer.convert_tokens_to_ids(tokens)
            """
            from kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1',
                                                        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6,
                                                                         'enable_sampling': True})
            self.tokenizer = tokenizer._tokenize
            self.encode = tokenizer.encode

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


