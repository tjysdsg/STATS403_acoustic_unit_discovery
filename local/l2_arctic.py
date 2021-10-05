"""
Based on https://github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/local/l2arctic_prep.py
"""
import glob
import os
import textgrid
import argparse
from typing import List, Tuple

EMPTY_PHONES = [
    '<blank>',
    '<unk>',
    'SPN',
    'SIL',
    '<sos/eos>',
]

# some files are broken
# YDCK/annotation/arctic_a0209.TextGrid -> YDCK-arctic_a0209
ERROR_UTTS = ["YDCK-arctic_a0209", "YDCK-arctic_a0272"]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare L2 data")
    parser.add_argument("--l2-path", help="l2-Arctic path")
    parser.add_argument("--output-dir", help="l2-Arctic path")
    return parser.parse_args()


def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1, len(phn_lst)):
        if phn_lst[i] == phn_lst[i - 1] and phn_lst[i] == "SIL":
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp


def clean_phone(phone: str):
    phone = phone.strip(" ").upper()
    if phone == "SP" or phone == "SIL" or phone == "" or phone == "SPN":
        ret = "SIL"
    else:
        if phone == "ER)":
            ret = "ER"
        elif phone == "AX" or phone == "AH)":
            ret = "AH"
        elif phone == "V``":
            ret = "V"
        elif phone == "W`":
            ret = "W"
        else:
            ret = phone

    return ret


def phone_to_score_phone(phone: str) -> str:
    if phone in EMPTY_PHONES:
        return phone

    if '*' in phone:
        return f'{phone.strip("*")}1'

    return f'{phone}2'


def get_utt_id(spk: str, filename: str):
    assert '.' not in filename  # filename must not contain a file extension
    return f'{spk}-{filename}'


def tokenize_utt_id(utt_id: str):
    """
    :return (spk_id, filename)
    """
    return utt_id.split('-')


def tokenize_path(path: str):
    path = path.replace('\\', '/')
    tokens = path.split("/")
    spk_id = tokens[-3]
    filename = tokens[-1].split('.')[0]
    return spk_id, filename


def build_data_path(prefix: str, spk_id: str, filename: str, annotation_type: str):
    """
    annotation_type can be one of ['textgrid', 'annotation', 'transcript', 'wav'],
    and the file extension is set accordingly
    """
    ext = 'TextGrid'
    if annotation_type == 'transcript':
        ext = 'txt'
    elif annotation_type == 'wav':
        ext = 'wav'

    return os.path.join(prefix, spk_id, annotation_type, f'{filename}.{ext}')


def clean_textgrids(
        data_dir: str, spk_and_filenames: List[Tuple[str, str]], output_dir: str, exclude_utts=None,
        clean_ppl=True
):
    if exclude_utts is None:
        exclude_utts = set()

    wrd_text = open(os.path.join(output_dir, "words"), 'w')
    wavscp = open(os.path.join(output_dir, "wav.scp"), 'w')
    cpl = open(os.path.join(output_dir, "cpl.txt"), 'w')  # correct phones
    utt2spk = open(os.path.join(output_dir, "utt2spk"), 'w')
    if clean_ppl:
        ppl = open(os.path.join(output_dir, "text"), 'w')  # perceived phones

    all_utts = []
    for spk_filename in spk_and_filenames:
        spk_id, filename = spk_filename
        utt_id = get_utt_id(spk_id, filename)

        # PPL path
        all_utts.append(utt_id)
        if utt_id in exclude_utts or utt_id in ERROR_UTTS:
            continue

        if clean_ppl:
            phn_path = build_data_path(data_dir, spk_id, filename, 'annotation')
        else:
            phn_path = build_data_path(data_dir, spk_id, filename, 'textgrid')
        # wav path
        wav_path = build_data_path(data_dir, spk_id, filename, 'wav')
        # CPL path
        text_path = build_data_path(data_dir, spk_id, filename, 'transcript')

        ppl_phones = []
        cpl_phones = []
        tg = textgrid.TextGrid.fromFile(phn_path)
        for i in tg[1]:  # type: textgrid.Interval
            if i.mark == '':
                continue
            else:
                if clean_ppl:
                    cpl_ppl_type = i.mark.split(",")  # [CPL] or [CPL, PPL, error_type]
                    if len(cpl_ppl_type) == 1:  # no pronunciation error
                        ppl_phn = cpl_ppl_type[0]
                    else:
                        ppl_phn = cpl_ppl_type[1]

                    cpl_phn = cpl_ppl_type[0]
                else:
                    cpl_phn = i.mark

                # remove stress marker
                cpl_phn = ''.join(c for c in cpl_phn if not c.isdigit())
                if clean_ppl:
                    ppl_phn = ''.join(c for c in ppl_phn if not c.isdigit())

                # clean phone
                cpl_phones.append(clean_phone(cpl_phn))
                if clean_ppl:
                    ppl_phones.append(clean_phone(ppl_phn))

        # remove empty phones from CPL
        cpl_phones = [p for p in cpl_phones if p not in EMPTY_PHONES]

        # postprocessing for data with PPLs
        if clean_ppl:
            # remove repeated SIL and convert to score-phones for PPL
            ppl_phones = del_repeat_sil(ppl_phones)
            ppl_phones = [phone_to_score_phone(p) for p in ppl_phones]

        f = open(text_path, 'r')
        for line in f:
            wrd_text.write(utt_id + " " + line.lower() + "\n")

        wavscp.write(f'{utt_id}\t{wav_path}\n')
        cpl.write(f'{utt_id}\t{" ".join(cpl_phones)}\n')
        utt2spk.write(f'{utt_id}\t{spk_id}\n')
        if clean_ppl:
            ppl.write(f'{utt_id}\t{" ".join(ppl_phones)}\n')

    wrd_text.close()
    wavscp.close()
    cpl.close()
    utt2spk.close()
    if clean_ppl:
        ppl.close()

    return all_utts


def main():
    args = get_args()
    data_dir = args.l2_path

    speakers = [
        "EBVS", "ERMS", "HQTV", "PNV", "ASI", "RRBI", "BWC", "LXC", "HJK", "HKK", "ABA", "SKA", "MBMPS", "THV",
        "SVBI", "NCC", "YDCK", "YBAA", "NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"
    ]
    print(f"n_speakers = {len(speakers)}")

    # test data
    path_tokens = []
    for spk in speakers:
        path = f"{data_dir}/{spk}/annotation/*.TextGrid"
        path_tokens += glob.glob(path)
    path_tokens = [tokenize_path(e) for e in path_tokens]
    output_dir = os.path.join(args.output_dir, 'l2arctic_test')
    os.makedirs(output_dir, exist_ok=True)

    test_utts = clean_textgrids(data_dir, path_tokens, output_dir, clean_ppl=True)

    # train data don't have PPL annotations
    path_tokens = []
    for spk in speakers:
        path = f"{data_dir}/{spk}/textgrid/*.TextGrid"
        path_tokens += glob.glob(path)
    path_tokens = [tokenize_path(e) for e in path_tokens]
    output_dir = os.path.join(args.output_dir, 'l2arctic_train')
    os.makedirs(output_dir, exist_ok=True)
    clean_textgrids(data_dir, path_tokens, output_dir, exclude_utts=set(test_utts), clean_ppl=False)


if __name__ == '__main__':
    main()
