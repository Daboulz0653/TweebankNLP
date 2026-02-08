import stanza
import json
# import pandas as pd
import logging
import sys
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger()


INPUT = "../combined_corpus.ndjson"
OUTPUT = "../combined_corpus_with_deps_identitylemma_withpretrain.ndjson"
BATCH_SIZE = 100


def configure_model():
    # config for the `en_tweet` models (models trained only on Tweebank)
    config = {
        'processors': 'tokenize,lemma,pos,depparse',
        'lang': 'en',

        'tokenize_pretokenized': False, # disable tokenization
        'tokenize_model_path': './twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',

        'lemma_use_identity': True,
        # 'lemma_model_path': './twitter-stanza/saved_models/lemma/en_tweet_lemmatizer.pt',

        "pos_model_path": './twitter-stanza/saved_models/pos/en_tweet_tagger.pt',
        "pos_pretrain_path": './twitter-stanza/saved_models/pos/en_tweet.pretrain_resaved.pt',

        "depparse_model_path": './twitter-stanza/saved_models/depparse/en_tweet_parser.pt',
        "depparse_pretrain_path": './twitter-stanza/saved_models/depparse/en_tweet.pretrain_resaved.pt',

        #"ner_model_path": './twitter-stanza/saved_models/ner/en_tweet_nertagger.pt',
    }

    # Initialize the pipeline using a configuration dict
    stanza.download("en")
    nlp = stanza.Pipeline(**config)

    return nlp


def serialize(doc):
    tree = []

    for sentence in doc.sentences:
        for word in sentence.words: #using words instead of tokens ---spacy does not take into account MWTs (?)
            word_info  = {
                'text': word.text,
                'lemma': word.lemma,
                'pos': word.upos,
                'tag': word.xpos if word.xpos is not None else word.upos,
                "dep": word.deprel,  # dependency relation
                "head_text": sentence.words[word.head - 1].text if word.head > 0 else "ROOT",  # text of the head token
                "head_idx": word.head - 1 if word.head > 0 else -1,  # index of the head token, -1 if no parent
                "idx": word.id - 1,  # token's own index, stanza begins at 1, so root is really 0
                "is_root": word.deprel == "ROOT",
                'children': [child.id - 1 for child in sentence.words if child.head == word.id]  # indices of children
            }
            tree.append(word_info)

    return tree

#do I need this for stanza? what is the purpose of this
def compact_structure(doc) -> Dict:
    sentences_triples = []

    for sent in doc.sentences:
        triples = []
        # root_indices = []

        for head, rel, dep in sent.dependencies:
            # if rel.lower() == "root" or dep.head == 0:
            #     root_indices.append(dep.id)
            #     continue

            triples.append({
                "head": head.id,
                "head_text": head.text,
                "rel": rel,
                "dep": dep.id,
                "dep_text": dep.text
            })

        sentences_triples.append(triples)
        # root_indices_per_sentence.append(root_indices)

    return {
        "sentences_triples": sentences_triples,
        # "root_indices": root_indices_per_sentence
    }


def process_batch(batch, nlp):
    results = []

    in_docs = [stanza.Document([], text=d) for d in batch] # Wrap each document with a stanza.Document object
    out_docs = nlp(in_docs)

    for doc in out_docs:
        parsed_data = {
           "full_tree": serialize(doc),
           "compact": compact_structure(doc),
           "num_tokens" : doc.num_tokens,
           "num_words" : doc.num_words,
           "num_sentences": len(doc.sentences)
        }
        results.append(parsed_data)
    return results

def count_lines(file):
    logger.info("counting lines in input file...")
    return sum(1 for _ in file)

def main():

    #---------------------------------------LOADING MODEL----------------------------------------
    logger.info("Loading Tweebank V2 Stanza Model")
    try:
        nlp = configure_model()
    except OSError as e:
        logger.error("Failed to configure Stanza model:", e)
        sys.exit(1)



    with open(INPUT, 'r', encoding='utf-8') as infile, \
        open(OUTPUT, 'w', encoding='utf-8') as outfile:

        errors = 0
        batch = []
        batch_entries = []

        # total_lines = count_lines(infile)
        # logger.info(f"total entries to process: {total_lines}")

        infile.seek(0)  # reset file ptr to beginning
        with tqdm(desc="processing", unit="entries") as progress_bar:
            for line in infile:
                try:
                    entry = json.loads(line)

                    if entry.get("type") == "submission":
                        # combine title and selftext for submissions
                        text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
                    else:  # comment
                        text = entry.get("body", "").strip()

                    if not text:
                        # skip empty entries
                        entry["dependency_parse"] = None
                        outfile.write(json.dumps(entry) + '\n')
                        progress_bar.update(1)
                        continue

                    batch.append(text)
                    batch_entries.append(entry)

                    if len(batch) >= BATCH_SIZE:
                        parsed_results = process_batch(batch, nlp)

                        for entry, parsed_data in zip(batch_entries, parsed_results):
                            entry["dependency_parse"] = parsed_data
                            outfile.write(json.dumps(entry) + '\n')
                        progress_bar.update(len(batch))
                        logger.info(f"total entries written to file: {len(batch)}")

                        batch = []
                        batch_entries = []

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    errors += 1
                    progress_bar.update(1)
                except Exception as e:
                    logger.warning(f"error processing entry: {e}")
                    errors += 1
                    progress_bar.update(1)

            #---------------------LEFTOVERS IN BATCH FINAL RUN--------------------
            if batch: #essentially checks if batch is empty
                try:
                    parsed_results = process_batch(batch, nlp)
                    for entry, parsed_data in zip(batch_entries, parsed_results):
                        entry["dependency_parse"] = parsed_data
                        outfile.write(json.dumps(entry) + '\n')
                    progress_bar.update(len(batch))
                except Exception as e:
                    logger.error(f"error processing final batch: {e}")
                    errors += len(batch)

    logger.info(f"errors: {errors:,}")
    logger.info(f"total entries processed: {progress_bar.n}")
    logger.info(f"output saved to: {OUTPUT}")

    logger.info("\nexample dependency parse structure:")
    with open(OUTPUT, 'r', encoding='utf-8') as f:
        first_entry = json.loads(f.readline())
        if first_entry.get("dependency_parse"):
            logger.info(f"  - full tree tokens: {first_entry['dependency_parse']['num_tokens']}")
            logger.info(f"  - sentences: {first_entry['dependency_parse']['num_sentences']}")
            logger.info(f"  - dependency triples: {len(first_entry['dependency_parse']['compact'])}")


if __name__ == "__main__":
    main()