## SenseVoice-python with onnx

[「简体中文」](./README.md)|「English」

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice) is an audio foundation model 
with audio understanding capabilities, 
including Automatic Speech Recognition (ASR), Language Identification (LID), 
Speech Emotion Recognition (SER), and Acoustic Event Classification (AEC) 
or Acoustic Event Detection (AED). 

Currently, SenseVoice-small supports multilingual speech recognition, emotion recognition, 
and event detection for Chinese, Cantonese, English, Japanese, and Korean, with extremely low inference latency.
This project provides the necessary ONNX environment installation 
and inference methods for the Python version of the SenseVoice model.

## Usage

### Installation 

```bash
pip install sensevoice-onnx

# or pip from GitHub
pip install git+https://github.com/lovemefan/SenseVoice-python.git
```

### Usage

```bash
sensevoice --audio sensevoice/resource/asr_example_zh.wav
```

```
Sense Voice Script Parameters

optional arguments:
  -h, --help            show this help message and exit
  -a , --audio_file     set the audio file path
  -dp , --download_path set the custom model download path, default is `sensevoice/resource`
  -d , --device         specify device, use -1 for CPU, specify card number for GPU (requires onnxruntime-gpu), default is `-1`
  -n , --num_threads    number of threads, default is `4`
  -l , --language {auto,zh,en,yue,ja,ko,nospeech}  language code, default is `auto`
  --use_itn             whether to use ITN
  --use_int8            whether to use int8 quantized ONNX model
```

### Result

```bash
2024-07-19 07:22:40,643 INFO [sense_voice_ort_session.py:130] Loading model from /home/runner/work/SenseVoice-python/SenseVoice-python/sensevoice/resource/embedding.npy
2024-07-19 07:22:40,647 INFO [sense_voice_ort_session.py:133] Loading model /home/runner/work/SenseVoice-python/SenseVoice-python/sensevoice/resource/sense-voice-encoder.onnx
2024-07-19 07:22:42,755 INFO [sense_voice_ort_session.py:140] Loading /home/runner/work/SenseVoice-python/SenseVoice-python/sensevoice/resource/sense-voice-encoder.onnx takes 2.11 seconds
2024-07-19 07:22:42,786 INFO [sense_voice.py:76] Audio sensevoice/resource/asr_example_zh.wav is 5.58 seconds
2024-07-19 07:22:43,102 INFO [sense_voice.py:81] [0.61s - 5.53s] <|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型
2024-07-19 07:22:43,102 INFO [sense_voice.py:83] Decoder audio takes 0.31638407707214355 seconds
2024-07-19 07:22:43,103 INFO [sense_voice.py:84] The RTF is 0.05669965538927304.
```