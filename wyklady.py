from pydub import AudioSegment, effects
from openai import OpenAI
from whisper.utils import get_writer
import whisper
import os

from time import localtime, strftime

# filename = "Renesans 2.m4a"
# part_file_name = "Renesans_2_part_1"
filename = "Sztuka renesansu w PL.m4a"
full_file_name = "Sztuka renesansu w PL"
part_file_name = "Sztuka_renesansu_w_PL_part"
time_in_minutes = 'full'
# output_directory = "./" + full_file_name + "_results"
output_directory = full_file_name + "_results"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

audio = AudioSegment.from_file(filename)
normalized_audio = effects.normalize(audio)

one_minute = 60 * 1000
test_first_x_min = normalized_audio  # [:one_minute*time_in_minutes]

export_format = 'mp3'
export_file_name_part = part_file_name + "_" + str(time_in_minutes)
export_file_name_full = full_file_name
export_file_name = export_file_name_full
export_file_path = output_directory + "/" + export_file_name

test_first_x_min.export(export_file_name + '.' + export_format, format=export_format)

# whisper api test

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))


selected_model = "large-v2"
selected_language = 'pl'
model = whisper.load_model(selected_model)
result = model.transcribe(export_file_name + '.' + export_format, language=selected_language, fp16=False, verbose=True, word_timestamps=True)
print(result["text"])
# print(result["segments"])

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
with open(export_file_path + '_' + selected_model + '_transcript.txt', 'w', encoding="utf-8") as f:
    f.write(result["text"])

srt_writer = get_writer("srt", "./" + output_directory)
srt_writer(result, export_file_name)

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("no elo")
# open api test - remote solution
"""
client = OpenAI()
audio_file= open("Renesans_2_part_1.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text",
  language="pl"
)
print(transcript)

with open('open_ai_transcript.txt', 'w') as f:
   f.write(transcript) """
