import time
import random
import argparse
import torch
import os
import numpy as np
from StyleTTS2.styletts2 import StyleTTS2
from scipy.io import wavfile
import json

def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)

    # Argument Parser Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation")
    parser.add_argument("--test-data", type=str, default="Data/Test_tts/test_data.json")
    parser.add_argument("--wav_save_path", type=str, default="TTS_results/ED_test/")
    args = parser.parse_args()

    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    for num, test_dia in enumerate(test_data):
        wav_save_path = args.wav_save_path + f'test{num+1}'
        if not os.path.exists(wav_save_path):
            os.makedirs(wav_save_path)
            print(f"Folder '{wav_save_path}' created.")
            continue_outer = False
        else:
            print(f"Folder '{wav_save_path}' already exists.")
            continue_outer = True
        if continue_outer:
            continue
        
        #TTS for response text
        tts = StyleTTS2()
        agent_timbre_tone = test_dia["Agent Timbre and Tone"]
        agent_gender = test_dia["Agent Gender"]
        empathetic_response = test_dia["Empathetic Response"]
        if agent_gender=='Female':
            wav_file = "StyleTTS2/Demo/reference_audio/W/" + agent_timbre_tone.lower() + '.wav'
        elif agent_gender=='Male':
            wav_file = "StyleTTS2/Demo/reference_audio/M/" + agent_timbre_tone.lower() + '.wav'

        result_name = wav_file.split('/')[-1]
        start = time.time()
        noise = torch.randn(1, 1, 256).to(args.device)
        ref_s = tts.compute_style(wav_file)
        wav = tts.inference(empathetic_response, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")

        scaled_data = np.int16(wav * 32767)
        wav_file_path = wav_save_path + '/' + result_name
        wavfile.write(wav_file_path, 24000, scaled_data)
        print(f"Saved as {wav_file_path}")

if __name__ == "__main__":
    main()