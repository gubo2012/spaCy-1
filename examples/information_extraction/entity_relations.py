#!/usr/bin/env python
# coding: utf8
"""A simple example of extracting relations between phrases and entities using
spaCy's named entity recognizer and the dependency parse. Here, we extract
money and currency values (entities labelled as MONEY) and then check the
dependency tree to find the noun phrase they are referring to – for example:
$9.4 million --> Net income.

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import spacy


TEXTS = [
    'Net income was $9.4 million compared to the prior year of $2.7 million.',
    'Revenue exceeded twelve billion dollars, with a loss of $1b.',
    'Net revenue for CEC was flat at $1.0 billion. On a same-store basis, which excludes the recently deconsolidated Horseshoe Baltimore from both years, net revenue was up 3.8% to $939 million, driven by strong gaming volume, hotel performance, and incremental revenues from operational initiatives.',
    'Net loss for CEC, before adjusting for noncontrolling interest, was $460 million, driven by an adjustment of $472 million to the restructuring of CEOC.',
    'Income from operations for CEC improved $130 million year-over-year to $86 million, representing an operating margin of 8.7%, due to accelerated stock based compensation of $145 million associated with the sale of the CIE social and mobile games business in the third quarter of 2016.',
    'Adjusted EBITDA for CEC improved $34 million, 12.6% year-over-year to $303 million, driving margins up 345 basis points to 30.7%. On a same-store basis, adjusted EBITDA improved $44 million, 17.7%, lifting margins 369 basis points to 31.2%.',
    'Repriced and refinanced debt through Q3 2017 that will reduce total annual interest expense by $270 million.'
]


@plac.annotations(
    model=("Model to load (needs parser and NER)", "positional", None, str))
def main(model='en_core_web_sm'):
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
    print("Processing %d texts" % len(TEXTS))

    for text in TEXTS:
        doc = nlp(text)
        relations = extract_currency_relations(doc)
        for r1, r2 in relations:
            print('{:<10}\t{}\t{}'.format(r1.text, r2.ent_type_, r2.text))


def extract_currency_relations(doc):
    # merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()

    relations = []
    for money in filter(lambda w: w.ent_type_ == 'MONEY', doc):
        if money.dep_ in ('attr', 'dobj'):
            subject = [w for w in money.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == 'pobj' and money.head.dep_ == 'prep':
            relations.append((money.head.head, money))
    return relations


if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # Net income      MONEY   $9.4 million
    # the prior year  MONEY   $2.7 million
    # Revenue         MONEY   twelve billion dollars
    # a loss          MONEY   1b
