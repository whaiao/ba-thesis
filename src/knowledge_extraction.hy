(import [collections [defaultdict]]
        pickle
        [pprint [pprint]]
        re
        [typing [List Dict Callable]]
        [bert-score [BERTScorer]]
        [numpy [ndarray]]
        [pandas :as pd]
        torch
        [src.data.atomic [Relation]]
        [src.constants [DATA-ROOT]]
        [src.nlp [srl dependency-parse NLP SemanticRoleLabel]]
        [src.utils [read-tsv]])

; constants
(setv Dataframe pd.DataFrame
      lookup (with [f (open f"{DATA-ROOT}/atomic/lookup.pickle" "rb")] (.load pickle f))
      atomic (read-tsv f"{DATA-ROOT}/atomic/processed.tsv")
      ;scorer (BERTScorer :lang "en" :rescale-with-baseline True)) ; this is roberta
      scorer (BERTScorer :model-type "distilbert-base-uncased" :lang "en" :rescale-with-baseline True))

(defn ^(. List [str]) extract-phrases 
  [^(. List [SemanticRoleLabel]) srl-parses]

  (defn ^str process [^str phrase]
    "Collects all tagged tokens and joins them"
    (setv matches (.join " " (.findall re r"\[\w+:[\s\w+]+\]" phrase re.IGNORECASE))
          matches (.sub re r"\[\w+:\s" "" matches)
          matches (.replace matches "]" ""))
    (return matches))

  (setv phrases []
        verbs [])

  ; extract verbs with the phrases
  (for [p srl-parses]
    (setv parse (process (. p parse)))
    (when (> (len (.split parse)) 2)
      (.append phrases parse)
      (.append verbs (. p verb))))
  (return [verbs phrases]))

(defn ^(. List [str]) search 
  [^str query
   ^str [matching-strategy "verbs"]]
  "Searches query in ATOMIC parse lookup table"

  (when (= matching-strategy "verbs")
    (setv query (. (get (NLP query) 0) lemma_)
          data (dfor [k v] (.items lookup)
                     :if (in query (:verbs v))
                     [k (:text v)])))
  (return data))

(defn ^(. List [dict]) extract-from-atomic 
  [^(. List [str]) candidates]
  "Extracts knowledge relations from ATOMIC
        Args:
            extract-from-atomic - list of queries to extract from ATOMIC
    "

  (setv res [])

  (for [candidate candidates]
    (setv data (py "atomic[atomic['head'] == candidate]")
          entries (defaultdict list))
    (for [[_ row] (.iterrows data)]
      (setv relation (:relation row)
            relation-value (get row "relation-text"))
      (.append (get entries relation) relation-value))
    (.append res {candidate entries}))

  (return res))

(defn ^(. List [str]) retrieve-overlap 
  [^str x
   ^Callable [srl-model srl]
   ^str [matching-strategy "verbs"]]
  "Creates overlap between selected strategy and other stuff
    Args:
      x - input sentence
      srl-model - model for srl parsing
      matching-strategy - what candidates to retrieve can be:
        - verbs
        - objects
        - parses
    "
  (setv parses (srl-model x)
        [verbs phrases] (extract-phrases parses)
        res [])

  (when (= matching-strategy "verbs")
    (for [(, verb phrase) (zip verbs phrases)]
      (setv search-results (dfor [k v] (.items (search :query verb)) [(. k text) v])
            ; working with placeholders vs. substituted strings?
            candidates (list (.keys search-results))
            n-cand (len candidates)
            padded-reference (lfor i (range n-cand) phrase)
            ;; precision recall f1
            [P R F] (.score scorer :refs padded-reference
                                   :cands candidates)
            amax (.argmax torch F :dim -1))
      (.append res (get search-results (get candidates amax)))))
  (return res))


; testing area
; (setv query "Where have you last seen it?"
;   res (extract-from-atomic (retrieve-overlap query)))
; (print f"The query for following knowledge structure is: {query}")
; (pprint res)

