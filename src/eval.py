from typing import List, Union
import bert_score


def get_bert_score(references: Union[List[str], str],
                   candidates: Union[List[str], str]) -> tuple:
    return bert_score.score(cands=candidates,
                            refs=references,
                            lang='en',
                            use_fast_tokenizer=True)
