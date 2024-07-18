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
Sense Voice 脚本参数设置

options:
  -a , --audio_file 设置音频路径
                        
  -e , --embedding 设置embedding, 默认 `sensevoice/resource/embedding.npy`
  
  -m , --model  onnx模型路径， 默认`sensevoice/resource/sense-voice-encoder.onnx`
                        
  -b , --bpe_model bpe文件路径，默认`sensevoice/resource/chn_jpn_yue_eng_ko_spectok.bpe.model`
                        
  -c , --cmvn_file cmnv文件路径，默认`sensevoice/resource/am.mvn`
                         
  -d , --device device id，使用cpu时为-1，使用gpu（需要安装onnxruntime-gpu）时指定卡号 默认`-1`
                        
  -n , --num_threads 线程数, 默认 `4`
                        
  -l , --language {auto,zh,en,yue,ja,ko,nospeech} 语音代码，默认`auto`
                        
  --use_itn    是否使用itn

```


```
2024-07-18 16:23:09,153 INFO [sense_voice_ort_session.py:129] Loading model from sensevoice/resource/embedding.npy
2024-07-18 16:23:09,158 INFO [sense_voice_ort_session.py:132] Loading model sensevoice/resource/sense-voice-encoder.onnx
<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型

```
