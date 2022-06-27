TEST_CORPUS = [
    """<|python|> def fibRec(n):
    if n < 2:
        return n
    else:
        return fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
    "Det här är ett test.",
    "Två meningar!",
    "12345",
    "4    3   2  10",
    "1  2   3    4     5",
    "Let's test this tokenizer.",
    'ℌej Hej --- TVÅ TVÅ TVÅ',
    "Exempel med siffror: 1234.5 plus 2^5.",
    "Exempel med whitespace: ett två  tre   fyra    .",
]

TEST_EXAMPLES = [
    # "Let's test this tokenizer.",
    """<|python|> def fibRec(n):
    if n < 2:
        return n
    else:
        return fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
    'ℌej Hej --- TVÅ TVÅ TVÅ',
    'ℌej Hej --- TVÅ TVÅ TVÅ --- Ç Ç',
    "Exempel ett.",
    "Exempel med siffror: 1234.5 plus 2^5.",
    "Exempel med whitespace: ett två  tre   fyra    .",
    "\ndef main(args):\n    parameters = Parameters()\n    for i in range(5):\n        print(i)",
]
