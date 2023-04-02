TEST_CORPUS = [
    """<|python|> def fibRec(n):
    if n < 2:
        return n
    else:
        return fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
    """def fibRec(n):
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
    """<|python|> def fibRec(n):
    if n < 2:
        return n
    else:
        return fibRec(n-1) + fibRec(n-2)""",
    'Det var en fuktig, grå sommardag i slutet av juni.',
    'It was a humid, grey summer day at the end of June.',

    ######################################

    # "Let's test this tokenizer.",
    """<|python|> def fibRec(n):
    if n < 2:
        return n
    else:
        return fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
#    """<|python|> def fibRec(n):
#    if n < 2:
#\treturn n
#    else:
#\treturn fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
    # """def fibRec(n):
    # if n < 2:
    #     return n
    # else:
    #     return fibRec(n-1) + fibRec(n-2)<|endoftext|>ett""",
    '[sv] Det var en fuktig, grå sommardag i slutet av juni.',
    '[no/bo] Det var en fuktig, grå sommerdag i slutten av juni.',
    '[no/ny] Det var ein fuktig, grå sommardag/sumardag i slutten av juni.',
    '[da] Det var en fugtig, grå sommerdag i slutningen af juni.',
    '[is] Það var rakur, grár sumardagur í lok júní.',
    '[en] It was a humid, grey summer day at the end of June.',
    '[de] Es war ein feuchter, grauer Sommertag im späten Juni.',
    '[nl] Het was een vochtige, grauwe zomerdag aan het einde van juni.',
    # 'ℌej Hej --- TVÅ TVÅ TVÅ',
    # 'ℌej Hej --- TVÅ TVÅ TVÅ --- Ç Ç',
    # "Exempel ett.",
    "Exempel med siffror: 1234.5",
    # "Exempel med whitespace: ett två  tre   fyra    .",
    # "\ndef main(args):\n    parameters = Parameters()\n    for i in range(5):\n        print(i)",
]
