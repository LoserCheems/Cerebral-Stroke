import json
import os
import re
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

class SPTokenizer:
    def __init__(self, model_path: str):
        # 重新加载分词器
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # 开始/结束 token 的 id
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        self.role_special_token_expression = "|".join([re.escape(token) for token in role_special_tokens])

    def tokenize(self, s: str, encode_special_tokens=False):
        if encode_special_tokens:
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                t.append(s[match.start():match.end()])
                last_index = match.end()
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.Encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.Decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.Decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ 使用词汇表将单词（str）转换为id。"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为单词（str）。"""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        return self.sp_model.IdToPiece(index)


class Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, encode_special_tokens=False,
                 **kwargs):
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
        self.encode_special_tokens = encode_special_tokens
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                         encode_special_tokens=encode_special_tokens,
                         **kwargs)

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ 将词汇表作为字典返回 """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):
        """ 使用词汇表将单词（str）转换为id。"""
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为单词（str）。"""
        return self.tokenizer.convert_id_to_token(index)
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        保存词汇表和特殊标记文件到目录中。

        参数:
            save_directory (`str`):
                保存词汇表的目录。
            filename_prefix (`str`, *optional*):
                要添加到保存的文件名的可选前缀。

        返回:
            `Tuple(str)`: 保存的文件的路径。
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens
    
    def build_single_message(self, role, metadata, message):
        assert role in ["system", "user", "assistant"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens
    
    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)


    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。BERT序列具有以下格式:

        - 单个序列: `[CLS] X [SEP]`
        - 一对序列: `[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个ID列表。

        返回:
            `List[int]`: 带有适当的特殊标记的[input IDs](../glossary#input-ids)列表。
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        填充编码的输入（在左/右和预定义长度或批次中的最大长度）

        参数:
            encoded_inputs:
                标记化输入（`List[int]`）的字典或标记化输入（`List[List[int]]`）的批处理。
            max_length: 返回的列表的最大长度和可选的填充长度（见下文）。
                将通过考虑特殊标记来截断。
            padding_strategy: 用于填充的PaddingStrategy。

                - PaddingStrategy.LONGEST 填充到批处理中最长的序列
                - PaddingStrategy.MAX_LENGTH: 填充到最大长度（默认）
                - PaddingStrategy.DO_NOT_PAD: 不填充
                分词器填充边在self.padding_side中定义:

                    - 'left': 在序列的左侧填充
                    - 'right': 在序列的右侧填充
            pad_to_multiple_of: (optional) 如果设置，将序列填充到提供的值的倍数。
                这对于启用NVIDIA硬件上的Tensor Core特别有用，计算能力为`>= 7.5`（Volta）。
            return_attention_mask:
                (optional) 设置为False以避免返回注意力掩码（默认值：设置为模型特定值）

        返回:
            带有填充的标记化输入的字典。
        """
        # 从模型默认加载
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果不存在，则初始化注意力掩码。
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs