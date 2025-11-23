from spell_correction_system import AdvancedSpellChecker, CorpusService, Config

sc = AdvancedSpellChecker()
if not sc.load_from_cache():
    print('Loading corpus and training...')
    cs = CorpusService()
    corpus = cs.load_corpus(lambda m: print('  ', m))
    sc.train(corpus, lambda m: print('  ', m))
else:
    print('Loaded from cache successfully')


def print_suggestions(word, ctx=''):
    print(f"\n--- {word} in context: {repr(ctx)}")
    print('  check_word:', sc.check_word(word))
    suggestions = sc.get_suggestions(word, ctx)
    if not suggestions:
        print('   (no suggestions)')
    for s in suggestions[:5]:
        print('   ->', s.corrected, '| dist=', s.edit_distance, '| conf=', round(s.confidence,3), '| source=', getattr(s, 'source', 'unknown'))


words = ['teh','speling','patiant','recieve','hosptial','too','their','then']

for w in words:
    ctx = 'The patient went to the hosptial' if w in ['teh','speling','patiant','recieve','hosptial'] else 'I will go too the store' if w=='too' else "Their going to the park" if w=='their' else 'He is better then me'
    print_suggestions(w, ctx)

print('\nSample words from vocabulary/dictionary:')
for w,f in sc.get_all_words_sorted()[:10]:
    print(' ', w, f)
