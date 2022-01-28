(import pickle
        re
        [typing [List Dict]]
        [bert-score [BERTScorer]]
        [numpy [ndarray]]
        [pandas :as pd]
        torch
        [src.constants [DATA-ROOT]]
        [src.nlp [srl dependency-parse NLP SemanticRoleLabel]]
        [src.utils [read-tsv]])

(setv Dataframe pd.DataFrame
      ATOMIC (with [f (open f"{DATA-ROOT}/atomic/lookup.pickle" "rb")] (.load pickle f))
      scorer (BERTScorer :lang "en" :rescale-with-baseline True))


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

(defn ^(. List [str]) search [^str query
              ^str [matching-strategy "verbs"]]
  "Searches query in ATOMIC parse lookup table"

  (when (= matching-strategy "verbs")
    (setv query (. (get (NLP query) 0) lemma_)
          data (dfor [k v] (.items ATOMIC)
                     :if (in query (:verbs v))
                     [k (:text v)])))
;          data (set (lfor candidate (.values ATOMIC) 
;                          :if (in query (:verbs candidate))
;                          (:text candidate)))))
  (return data))

(defn extract [^dict candidate]
  "Extracts knowledge relations from ATOMIC"
  (raise (NotImplementedError)))

(defn prepare-overlap [^str x
                       [srl-model None]
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
  (setv parses (srl x)
        [verbs phrases] (extract-phrases parses))

  ;; TODO: check this tomorrow -- if candidates are correctly supplied
  (when (= matching-strategy "verbs")
    (for [(, verb phrase) (zip verbs phrases)]
      (setv search-results (dfor [k v] (.items (search :query verb)) [(. k text) v])
            candidates (list (.keys search-results))
            n-cand (len candidates)
            padded-reference (lfor i (range n-cand) phrase)
            [P R F] (.score scorer :refs padded-reference
                                   :cands candidates)
            amax (.argmax torch F :dim -1))
      (print candidates)
      (print (get search-results (get candidates amax))))))

(prepare-overlap "i stole a bike and bought a teddy bear")
