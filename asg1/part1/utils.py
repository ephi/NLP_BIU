import re


def create_word_signature(word):
    naun_suffix = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ty', 'ity',
                   'ment', 'ness', 'ship', 'tion', 'sion']
    verb_suffix = ['ate', 'en', 'fy', 'ify', 'ize']
    adgective_suffix = ['able', 'ible', 'al', 'ful', 'ic', 'ical', 'ous', 'ish', 'ive', 'less']
    general_suffix = ['ing', 'ed', 'ure', 'age', 'ages']
    word = word.lower()
    for suffix in naun_suffix + verb_suffix + adgective_suffix + general_suffix:
        if word[-len(suffix):] == suffix:
            return f'^{suffix.upper()}'

    # Some general patterns:
    if re.search(r'^[0-9]+[,/.][0-9]+[,]?[0-9]*$', word) is not None:
        return '^NUM'
    if re.search(r'^[0-9]+:[0-9]+$', word) is not None:
        return '^HOUR'
    if re.search(r'^[0-9]+/[0-9]+-[a-zA-Z]+[-]?[a-zA-Z]*$', word) is not None:
        return '^FRUC-WORD'
    if re.search(r'^[A-Z]+-[A-Z]+$', word) is not None:
        return '^A-A'
    if re.search(r'^[a-z]+-[a-z]+$', word) is not None:
        return '^a-a'
    if re.search(r'^[A-Z][a-z]+-[A-Z][a-z]+$', word) is not None:
        return '^Aa-Aa'
    if re.search(r'^[A-Z]+$', word) is not None:
        return '^UPPER_CASE'
    if re.search(r'^[A-Z][a-z]+$', word) is not None:
        return '^Aa'

    return '^UNK'
