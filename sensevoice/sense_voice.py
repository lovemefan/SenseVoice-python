# -*- coding:utf-8 -*-
# @FileName  :sense_voice.py.py
# @Time      :2024/7/18 15:40
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse

from sensevoice.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
from sensevoice.utils.frontend import WavFrontend

languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Sense Voice")
    arg_parser.add_argument("-a", "--audio_file", required=True, type=str, help="Model")
    arg_parser.add_argument(
        "-e",
        "--embedding",
        default="sensevoice/resource/embedding.npy",
        type=str,
        help="Embedding",
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        default="sensevoice/resource/sense-voice-encoder.onnx",
        type=str,
        help="Model",
    )
    arg_parser.add_argument(
        "-b",
        "--bpe_model",
        default="sensevoice/resource/chn_jpn_yue_eng_ko_spectok.bpe.model",
        type=str,
        help="BPE model",
    )
    arg_parser.add_argument(
        "-c", "--cmvn_file", default="sensevoice/resource/am.mvn", type=str, help="cmvn file"
    )
    arg_parser.add_argument("-d", "--device", default=-1, type=int, help="Device")
    arg_parser.add_argument(
        "-n", "--num_threads", default=4, type=int, help="Num threads"
    )
    arg_parser.add_argument(
        "-l",
        "--language",
        choices=languages.keys(),
        default="auto",
        type=str,
        help="Language",
    )
    arg_parser.add_argument("--use_itn", action="store_true", help="Use ITN")
    args = arg_parser.parse_args()

    front = WavFrontend(args.cmvn_file)

    model = SenseVoiceInferenceSession(
        args.embedding, args.model, args.bpe_model, args.device, args.num_threads
    )
    audio_input = front.get_features(args.audio_file)
    asr_result = model(
        audio_input[None, ...], language=languages[args.language], use_itn=args.use_itn
    )
    print(asr_result)
