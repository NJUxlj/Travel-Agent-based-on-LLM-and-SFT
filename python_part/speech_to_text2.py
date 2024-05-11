import sounddevice as sd
import numpy as np
import openai

# 设置你的OpenAI API 密钥
api_key = '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'
openai.api_key = api_key

# 录音参数
sample_rate = 16000  # 采样率
duration = 5  # 每次录音持续时间（秒）

def audio_callback(indata, frames, time, status):
    """这个回调函数在每次音频缓冲区填满时被调用"""
    if status:
        print(status)
    # 将numpy音频数组转换为字节数据
    audio_bytes = indata.tobytes()

    # 调用OpenAI的Speech to Text API
    try:
        response = openai.Audio.transcriptions.create(
            model="whisper-1",
            audio=audio_bytes,
            format="raw",
            sample_rate=sample_rate,
            channels=1
        )
        print("Transcribed Text:", response['text'])
    except Exception as e:
        print("Error:", e)

# 使用sounddevice库录制音频
with sd.InputStream(callback=audio_callback,
                    dtype='int16',
                    channels=1,
                    samplerate=sample_rate):
    print("Recording... Press Ctrl+C to stop.")
    sd.sleep(duration * 1000)  # 持续录音一定时间