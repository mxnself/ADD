from collections import Counter
from collections import defaultdict
import re
from tqdm import tqdm
import pickle
from unidecode import unidecode
import inflect


def get_phoneme_dict_symbols(unknown: str = "<unk>", eos: str = "~"):
    """
    :param unknown: 
    :param eos: 
    :return:
    """
    symbols = [
        'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
        'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
        'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
    phonemes = ['@' + s for s in symbols]
    symbols_list = [unknown, eos] + list(chars) + phonemes

    dict_set = {s: i for i, s in enumerate(symbols_list)}

    return dict_set, set(symbols)

def _clean_dict(cmu_dict_path):
    _, symbols_set = get_phoneme_dict_symbols()

    alt_re = re.compile(r'\([0-9]+\)')
    cmu_dict = {}
    p2w_dict ={}

    with open(cmu_dict_path, 'r', encoding='latin-1') as cmu_file:
        for line in cmu_file:
            if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
                parts = line.split('  ')
                word = re.sub(alt_re, '', parts[0])

                pronunciation = " "
                temps = parts[1].strip().split(' ')
                for temp in temps:
                    if temp not in symbols_set:
                        pronunciation = None
                        break
                if pronunciation:
                    pronunciation = ' '.join(temps)
                    if word in cmu_dict:
                        cmu_dict[word].append(pronunciation)
                    else:
                        cmu_dict[word] = [pronunciation]
                    if pronunciation in p2w_dict:
                        if word not in p2w_dict[pronunciation]:
                            p2w_dict[pronunciation].append(word)
                    else:
                        p2w_dict[pronunciation] = [word]


    return cmu_dict, p2w_dict

def text_to_phonemes_converter(text: str, cmu_dict):
    """
    :param text:
    :param cmu_dict_path:
    :return: 
    """

    text = _clean_text(text)
    text = re.sub(r"([?.!,])", r" \1", text)

    cmu_result = []
    for word in text.split(' '):

        cmu_word = cmu_dict.get(word.upper(), [word])[0]
        if cmu_word != word:
            cmu_result.append(cmu_word + " |")
        else: 
            cmu_result.append(cmu_word)
    return " ".join(cmu_result)

def _clean_number(text: str):
    """
    :param text: 
    :return: 
    """
    comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
    decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
    dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
    ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    number_re = re.compile(r"[0-9]+")

    text = re.sub(comma_number_re, lambda m: m.group(1).replace(',', ''), text)
    text = re.sub(pounds_re, r"\1 pounds", text)
    text = re.sub(dollars_re, _dollars_to_word, text)
    text = re.sub(decimal_number_re, lambda m: m.group(
        1).replace('.', ' point '), text)
    text = re.sub(ordinal_re, lambda m: inflect.engine(
    ).number_to_words(m.group(0)), text)
    text = re.sub(number_re, _number_to_word, text)

    return text


def _number_to_word(number_re: str):
    """

    :param number_re: 
    :return:
    """
    num = int(number_re.group(0))
    tool = inflect.engine()

    if 1000 < num < 3000:
        if num == 2000: 
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + tool.number_to_words(num % 100)
        elif num % 100 == 0:
            return tool.number_to_words(num // 100) + " hundred"
        else:
            return tool.number_to_words(num, andword="", zero='oh', group=2).replace(", ", " ")
    else:
        return tool.number_to_words(num, andword="")


def _dollars_to_word(dollars_re: str):
    """ 
    :param dollars_re:
    :return:
    """
    match = dollars_re.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _abbreviations_to_word(text: str):
    """
    :param text: 
    :return: 
    """
    abbreviations = [
        (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort')
            #,('\'s', 'is')  #########################################TODO How to expand the abbreviation???
        ]
    ]

    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def _clean_text(text: str):
    """
    :param text: 
    :return: 
    """
    text = unidecode(text)
    text = text.lower()
    text = _clean_number(text=text)
    text = _abbreviations_to_word(text=text)
    text = re.sub(r"\s+", " ", text)

    return text

def separate_and_save_phones(phones_file_path, phones_sorted_bucket, inner_sorted_bucket):
    words_set = []
    words_dict = defaultdict(list)
    _, symbols_set = get_phoneme_dict_symbols()
    with open(phones_file_path, 'r') as fr:
        datas = fr.readlines()

    for line in tqdm(datas, desc="Split phones"):
        line = line.split(' | ')
        words_set += line


    # Count the number of words by the prefix they start with
    for word in tqdm(words_set, desc="Merge and count phones"):
        if word == '\n':
            continue
        prefix = word.split()[0]
        if prefix in symbols_set:
            words_dict[prefix].append(word)

    sorted_words = sorted(words_dict.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_aims = []


    with open(phones_sorted_bucket, 'wb') as file:
        pickle.dump(sorted_words, file)
        print("Phone buckets dict is stored")


    for phone_dict in tqdm(sorted_words, desc="Sort inner elements in every phone"):
        phones = Counter(phone_dict[1])
        sorted_phones = sorted(phones, key=phones.get, reverse=True)
        sorted_aims.append(sorted_phones)
    with open(inner_sorted_bucket, 'wb') as file:
        pickle.dump(sorted_aims, file)
        print("Every items in phone buckets are stored")

def flatten_list(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten_list(element))
        else:
            result.append(element)
    return result

def remove_duplicates(my_list):
    seen = set()
    result = []
    i = 1
    for item in my_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
            i += 1
    return result

def trans_and_save_sp(sentencepiece_dict, cmu_dict, inner_sorted_bucket, sp_dict_save_path):
    recover_dict = {}
    word_phones = []
    order_dict = {}
    with open(sentencepiece_dict, 'r') as fr:
        datas = fr.readlines()

    for line in tqdm(datas, desc='Convert every word in sentencepiece'):
        word = line.split()[0]
        tmp = word.replace('▁', '')

        if tmp.upper() in cmu_dict:
            phone = cmu_dict[tmp.upper()][0]  
            word_phones.append(phone) 
        else:
            word_phones.append(word)
            continue

        if phone in recover_dict.keys():
            if word not in recover_dict[phone]:
                recover_dict[phone].append(word)
        else:
            recover_dict[phone] = [word]



    with open(inner_sorted_bucket, 'rb') as file:
        orders = pickle.load(file)

    orders = flatten_list(orders)
    order_dict = {element: index for index, element in enumerate(orders)}

    max_index = len(order_dict) + 1

    word_phones = [[word_phones[i], order_dict[word_phones[i]] if word_phones[i] in order_dict else max_index] for i in range(len(word_phones))]

    sorted_phones = sorted(word_phones, key=lambda x: x[1])


    words_orders = [recover_dict[phone[0]] if phone[0] in recover_dict else [phone[0]] for phone in sorted_phones]

    words_dict = remove_duplicates(flatten_list(words_orders))
    words_dict = [line + ' 1\n' for line in words_dict]

    with open(sp_dict_save_path, 'w') as fw:
        for line in tqdm(words_dict, desc='Clean dict saved'):
            fw.write(line)



# path of cmu
cmu_dict_path = "../data/phone/cmudict-0.7b"
# path of train wrd
words_file_path = "../data/phone/librispeech.train.wrd"
# save phones
phones_file_path = "../data/phone/phones.txt"
# sort
phones_sorted_bucket = "../data/phone/sorted_bucket_dict.pkl"
inner_sorted_bucket = "../data/phone/sorted_phones_dict.pkl"
# original dict
sentencepiece_dict = "../data/phone/ori_dict.txt"
# sorted dict
sp_dict_save_path = "../data/phone/sorted_phones.txt"


def main():
    cmu_dict, p2w_dict = _clean_dict(cmu_dict_path)
    words_file = open(words_file_path,"r")
    phones_file = open(phones_file_path,"w")   
    lines = words_file.readlines()
    lines_len = len(lines)
    for i in tqdm(range(lines_len), desc= "Convert words to phones in train set"):
        texts1 = str(lines[i])
        phones_file.write(str(text_to_phonemes_converter(texts1, cmu_dict)) + "\n")

    words_file.close()
    phones_file.close()
    print("Words to phones convertion done!")
    separate_and_save_phones(phones_file_path, phones_sorted_bucket, inner_sorted_bucket)
    trans_and_save_sp(sentencepiece_dict, cmu_dict, inner_sorted_bucket, sp_dict_save_path)


if __name__ == "__main__":
    main()
