import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import glob
import torch
import torch.nn.functional as F
import fairseq
import torchaudio
import csv
import os
from tqdm import tqdm

exclude_labels = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
    "Babbling",
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    "Outside, urban or manmade",
    "Outside, rural or natural",
    "Reverberation",
    "Echo",
    "Noise",
    "Environmental noise",
    "Static",
    "Distortion",
    "Sidetone",
    "Cacophony",
    "White noise",
    "Pink noise",
    "Radio",
    "Television",
    "Field recording",
    "Silence",
]

# global normalization for AudioSet as default (norm_mean=-4.268 && norm_std=4.569)
def get_parser():
    parser = argparse.ArgumentParser(
        description="use fine-tuned EAT for inference in downstream tasks"
    )
    parser.add_argument('--source_path', help='location of source wav files', required=True)
    parser.add_argument('--label_file', help='location of label files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for fine-tuned model', required=True)
    parser.add_argument('--target_length', type=int, help='the target length of Mel spectrogram in time dimension', required=True)
    parser.add_argument('--top_k_prediction', type=int, help='the number of top k classes prediction in inference', required=True)
    parser.add_argument('--norm_mean', type=float, help='mean value for normalization', default=-4.268)
    parser.add_argument('--norm_std', type=float, help='standard deviation for normalization', default=4.569)

    return parser

def build_dictionary(label_path):
    vocab = {}
    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = int(row[0])
            label = row[2]
            vocab[index] = label
    return vocab


@dataclass
class UserDirModule:
    user_dir: str
    


def main():
    parser = get_parser()
    args = parser.parse_args()
    # print(args)
    os.makedirs(args.save_path.rsplit('/',1)[0], exist_ok=True)
    source_path = args.source_path
    label_file = args.label_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    target_length = args.target_length
    top_k_prediction = args.top_k_prediction
    norm_mean = args.norm_mean
    norm_std = args.norm_std

    vocab = build_dictionary(label_file)
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()
    source_files = glob.glob(os.path.join(source_path,"*.wav"))
    save_list = []
    for source_file in tqdm(source_files):
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        source = torch.from_numpy(wav).float().cuda()
        if sr == 16e3:
            print("Original sample rate is already 16kHz in file {}".format(source_file))
        else: 
            source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=16000).float().cuda()
            print("It is resampled to 16kHz in file {}".format(source_file))
            
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        
        source = source - source.mean()
        source = source.unsqueeze(dim=0)
        source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
        
        n_frames = source.shape[1]
        diff = target_length - n_frames
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
            source = m(source)
            
        elif diff < 0:
            source = source[:,0:target_length, :]
                    
        source = (source - norm_mean) / (norm_std * 2)
        
        with torch.no_grad():
            try:
                source = source.unsqueeze(dim=0) #btz=1
                pred = model(source)
                pred = torch.sigmoid(pred)
                topk_values, topk_indices = torch.topk(pred, top_k_prediction)
                inference = {vocab[index.item()]: value.item() for index, value in zip(topk_indices[0], topk_values[0])}
                output = []
                for label,res in inference.items():
                    if res > 0.1 and label not in exclude_labels:
                        output.append(label)
                output = ", ".join(output)
                save_list.append(source_file.split("/")[-1].replace(".wav","")+"\t"+output)
            except:
                print("Error in inference from {}".format(source_file))
                Exception("Error in inference from {}".format(source_file))
                
    with open(args.save_path, 'w') as file:
        for item in save_list:
            file.write(f"{item}\n")


if __name__ == '__main__':
    main()
