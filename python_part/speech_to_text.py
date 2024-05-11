import sounddevice as sd
import numpy as np
import whisper

import os


# 代理地址
# os.environ["OPENAI_API_BASE"] = "http://{your proxy url}/v1"
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'



# 加载Whisper模型
model = whisper.load_model("base")

def audio_callback(indata, frames, time, status):
    """这个回调函数在每次音频缓冲区填满时被调用"""
    if status:
        print(status)
    # 将音频数据转换为Whisper模型需要的格式
    audio = np.int16(indata * 32768).flatten()

    # 使用Whisper模型进行语音识别
    result = model.transcribe(audio)
    print("Transcribed Text:", result['text'])

# 录音参数
sample_rate = 16000  # 采样率
duration = 5  # 每次录音持续时间（秒）

# 使用sounddevice库录制音频
with sd.InputStream(callback=audio_callback,
                    dtype='float32',
                    channels=1,
                    samplerate=sample_rate):
    print("Recording... Press Ctrl+C to stop.")
    sd.sleep(duration * 1000)  # 持续录音一定时间