import hazm
# Persian Text Preprocessing
def build_persian_vocab():
    # Persian alphabet + common punctuation
    persian_chars = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآةئؤء،؟!. " + "".join(str(i) for i in range(10))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(persian_chars)}  # 0 for padding
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def text_to_sequence(text, char_to_idx):
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)  # Normalize Persian text
    return [char_to_idx.get(char, 0) for char in text]