## SenseVoice-python with onnx

「简体中文」|「[English](./README-EN.md)」


[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)是具有音频理解能力的音频基础模型，
包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。

当前SenseVoice-small支持中、粤、英、日、韩语的多语言语音识别，情感识别和事件检测能力，具有极低的推理延迟。
本项目提供python版的SenseVoice模型所需的onnx环境安装的与推理方式。


## 使用方式

### 安装
```bash
pip install sensevoice-onnx

# or pip from github
pip install git+https://github.com/lovemefan/SenseVoice-python.git
```

### 使用

```bash
sensevoice --audio sensevoice/resource/asr_example_zh.wav
```


第一次使用会自动从huggingface下载，如果下载不下来，可以使用镜像

* Linux:
```bash 
export HF_ENDPOINT=https://hf-mirror.com
```

* Windows Powershell

```bash
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

或者非入侵方式使用环境变量
```bash
HF_ENDPOINT=https://hf-mirror.com sensevoice --audio sensevoice/resource/asr_example_zh.wav
```



```
Sense Voice 脚本参数设置

optional arguments:
  -h, --help            show this help message and exit
  -a , --audio_file 设置音频路径
  -dp , --download_path 自定义模型下载路径，默认`sensevoice/resource`
  -d , --device, 使用cpu时为-1，使用gpu（需要安装onnxruntime-gpu）时指定卡号 默认`-1`
                        Device
  -n , --num_threads , 线程数, 默认 `4`
                        Num threads
  -l , --language {auto,zh,en,yue,ja,ko,nospeech} 语音代码，默认`auto`
  --use_itn             是否使用itn
  --use_int8            是否使用int8 量化的onnx模型

```

### 结果


```bash
2024-07-19 15:08:06,651 INFO [sense_voice_ort_session.py:130] Loading model from /Users/cenglingfan/Code/python-project/SenseVoice-python/sensevoice/resource/embedding.npy
2024-07-19 15:08:06,654 INFO [sense_voice_ort_session.py:133] Loading model /Users/cenglingfan/Code/python-project/SenseVoice-python/sensevoice/resource/sense-voice-encoder.onnx
2024-07-19 15:08:08,773 INFO [sense_voice_ort_session.py:140] Loading /Users/cenglingfan/Code/python-project/SenseVoice-python/sensevoice/resource/sense-voice-encoder.onnx takes 2.12 seconds
2024-07-19 15:08:08,802 INFO [sense_voice.py:76] Audio resource/asr_example_zh.wav is 5.58 seconds
2024-07-19 15:08:09,007 INFO [sense_voice.py:81] <|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型
2024-07-19 15:08:09,007 INFO [sense_voice.py:83] Decoder audio takes 0.20529699325561523 seconds
2024-07-19 15:08:09,007 INFO [sense_voice.py:84] The RTF is 0.0367915758522608.
```
