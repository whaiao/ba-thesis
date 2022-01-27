(import pickle
        re
        [typing [List]]
        [numpy [ndarray]]
        [pandas :as pd]
        [src.constants [DATA-ROOT]]
        [src.eval [get-bert-score]]
        [src.nlp [srl dependency-parse]]
        [src.utils [read-tsv]])

(setv Dataframe pd.DataFrame)

(defn ^(. List [str]) extract-phrases [^dict srl-parses]
  (defn process [^str phrase]
    (setv matches (.join " " (.findall re r"\[\w+:[\s\w+]+\]" phrase re.IGNORECASE))
          matches (.sub re r"\[\w+:\s" "" matches)
          matches (.replace matches "]" ""))
    (return matches))

  (setv phrases []
        verbs [])

  ; extract verbs with the phrases
  (for [(, k v) (.items srl-parses)]
    (for [parses v]
      (setv parse (process parses.parse))
      (when (> (len (.split parse)) 2)
        (.append phrases parse)
        (.append verbs parses.verb))))
  (return [verbs phrases]))


(defn prepare-overlap [^str x
                       ^Dataframe atomic
                       [srl-model None]
                       ^str [matching-strategy "verbs"]]
  "Creates overlap between selected strategy and other stuff
    Args:
      x - input sentence
      atomic - atomic dataframe to look up references
      srl-model - model for srl parsing
      matching-strategy - what candidates to retrieve can be:
        - verbs
        - objects
        - parses
    "
  (setv parses (srl-model x)
        phrases (extract-phrase parses))
  
  ; TODO: overlap filter verbs from dataframe
  (for [(, phrase verb) phrases]
    (print verb)))


; testing area
(setv path (+ DATA-ROOT "/social_chemistry/")
      data (with [p (open f"{path}srl.pickle" "rb")] (.load pickle p)))

(print (extract-phrases data))
