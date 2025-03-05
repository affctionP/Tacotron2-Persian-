import os
input_dirs=['./Dataset/',]
for input_dir in input_dirs:
    with open(os.path.join(input_dir, 'test.csv'), encoding='utf-8') as f:
        lines = f.readlines()  # Read all lines into a list
        for line in lines[1:]:
            parts = line.strip().split(',')
            basename = parts[1]
            wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
            text = parts[2]
            
            print(f"file : {basename} is {text}")