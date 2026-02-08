import stanza

# config for the `en_tweet` models (models trained only on Tweebank)
config = {
    'processors': 'tokenize,lemma,pos,depparse',
    'lang': 'en',
    'tokenize_pretokenized': False, # disable tokenization
    'tokenize_model_path': './twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',
    #'lemma_use_identity': True,
    # 'lemma_model_path': './twitter-stanza/saved_models/lemma/en_tweet_lemmatizer.pt',
    "pos_model_path": './twitter-stanza/saved_models/pos/en_tweet_tagger.pt',
    "depparse_model_path": './twitter-stanza/saved_models/depparse/en_tweet_parser.pt',
    #"ner_model_path": './twitter-stanza/saved_models/ner/en_tweet_nertagger.pt',
}

# Initialize the pipeline using a configuration dict
stanza.download("en")
nlp = stanza.Pipeline(**config)
doc = nlp("Thats why I absolutely hate this public hate against AI. \u201eOh you have AI show me how to disable it.\u201c stfu its so incredibly sad that many humans dont see the possibilities and beauty behind this new technology. After only 2 years its \u201eAI bad pfff\u201c wtf. Human nature can be so stupid sometimes")
print(doc) # Look at the result
