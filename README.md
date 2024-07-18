## SenseVoice-python with onnx

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)是阿里开源的多语言asr.



## 使用方式

下载checkpoint, 详见[huggingface](https://huggingface.co/lovemefan/SenseVoice-onnx)


将所有资源文件下载到sensevoice/resource文件夹下, 所有文件如下：

```
sensevoice/resource
├── am.mvn
├── asr_example_zh.wav
├── chn_jpn_yue_eng_ko_spectok.bpe.model
├── embedding.npy
├── sense-voice-encoder-int8.onnx
└── sense-voice-encoder.onnx

```


```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python sensevoice/sense_voice.py --audio sensevoice/resource/asr_example_zh.wav 
```


```
2024-07-18 16:23:09,153 INFO [sense_voice_ort_session.py:129] Loading model from sensevoice/resource/embedding.npy
2024-07-18 16:23:09,158 INFO [sense_voice_ort_session.py:132] Loading model sensevoice/resource/sense-voice-encoder.onnx
<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型

```
