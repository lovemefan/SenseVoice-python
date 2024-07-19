# -*- coding:utf-8 -*-
# @FileName  :sense_voice_onnxruntime.py
# @Time      :2024/7/17 20:53
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import sentencepiece as spm
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


class OrtInferRuntimeSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            device_id != "-1"
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)

        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            result = self.session.run(self.get_output_names(), input_dict)
            return result
        except Exception as e:
            print(e)
            raise RuntimeError(f"ONNXRuntime inferece failed. ") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


def log_softmax(x: np.ndarray) -> np.ndarray:
    # Subtract the maximum value in each row for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    # Calculate the softmax of x
    softmax = np.exp(x - x_max)
    softmax_sum = np.sum(softmax, axis=-1, keepdims=True)
    softmax = softmax / softmax_sum
    # Calculate the log of the softmax values
    return np.log(softmax)


class SenseVoiceInferenceSession:
    def __init__(
        self,
        embedding_model_file,
        encoder_model_file,
        bpe_model_file,
        device_id=-1,
        intra_op_num_threads=4,
    ):
        logging.info(f"Loading model from {embedding_model_file}")

        self.embedding = np.load(embedding_model_file)
        logging.info(f"Loading model {encoder_model_file}")
        start = time.time()
        self.encoder = OrtInferRuntimeSession(
            encoder_model_file,
            device_id=device_id,
            intra_op_num_threads=intra_op_num_threads,
        )
        logging.info(
            f"Loading {encoder_model_file} takes {time.time() - start:.2f} seconds"
        )
        self.blank_id = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_file)

    def __call__(self, speech, language: int, use_itn: bool) -> np.ndarray:
        language_query = self.embedding[[[language]]]

        # 14 means with itn, 15 means without itn
        text_norm_query = self.embedding[[[14 if use_itn else 15]]]
        event_emo_query = self.embedding[[[1, 2]]]

        input_content = np.concatenate(
            [
                language_query,
                event_emo_query,
                text_norm_query,
                speech,
            ],
            axis=1,
        ).astype(np.float32)
        input_length = np.array([input_content.shape[1]], dtype=np.int64)

        encoder_out = self.encoder((input_content, input_length))[0]

        def unique_consecutive(arr):
            if len(arr) == 0:
                return arr
            # Create a boolean mask where True indicates the element is different from the previous one
            mask = np.append([True], arr[1:] != arr[:-1])
            out = arr[mask]
            out = out[out != self.blank_id]
            return out.tolist()

        hypos = unique_consecutive(encoder_out[0].argmax(axis=-1))
        text = self.sp.DecodeIds(hypos)
        return text
