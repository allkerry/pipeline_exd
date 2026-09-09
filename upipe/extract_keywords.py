import yake, re, string, nltk
from exorde_data import Keywords, Translation

MAX_KEYWORD_LENGTH = 100

kw_extractor1 = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20)
kw_extractor2 = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, dedupFunc='seqm', windowsSize=7, top=10)

def filter_strings(lst):
    out = []
    special_chars = set(string.punctuation.replace("-",""))
    for s in lst:
        if not isinstance(s, str): continue
        length = len(s)
        ok = all(c=="-" or c not in special_chars for c in s) if length<=3 else all(c not in special_chars for c in s)
        if not ok: continue
        sc = sum(1 for c in s if c in string.punctuation or c.isnumeric() or not c.isalpha())
        s = re.sub('^[^A-Za-z0-9 ]+|[^A-Za-z0-9 ]+$','',s)
        if len(s)>0 and (sc*100/len(s))<=20 and any(c.isalpha() for c in s) and s not in out:
            out.append(s)
    return out

def remove_invalid(lst):
    out = []
    for s in lst:
        s = re.sub(r'//|https?:\/\/.*[\r\n]*','',s)
        if 2 < len(s) <= MAX_KEYWORD_LENGTH and s not in out: out.append(s)
    return out

def process_keywords(kws):
    out = []
    for k in kws:
        if k.isupper(): out.extend([k, k.lower()])
        elif not k.islower(): out.append(k.lower())
        else: out.append(k)
    return list(dict.fromkeys(out))

def extract_keywords(translation: Translation) -> Keywords:
    content = translation.translation
    kx = [e[0] for e in set(kw_extractor1.extract_keywords(content))]
    kx += [e[0] for e in set(kw_extractor2.extract_keywords(content))]
    kx = filter_strings(kx)
    kx = remove_invalid(kx)
    kx = process_keywords(kx)
    return Keywords(list(set(kx)))
