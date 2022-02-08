"hard-coding labels"
(import [pprint [pprint]]
        [datasets [load-dataset]]
        [datasets.dataset-dict [DatasetDict]]
        [transformers [AutoTokenizer]]
        torch)


;; init model, data and code labels
(defn ^DatasetDict empathetic-dialogues-emotion-binning []
  (setv tokenizer (.from-pretrained AutoTokenizer "benjaminbeilharz/bert-base-uncased-empatheticdialogues-sentiment-classifier")
        empathetic-dialogues (load-dataset "empathetic_dialogues")
        emo-set (set (get (:train empathetic-dialogues) "context"))
        mapped {"positive" ["caring"
                            "confident"
                            "content"
                            "excited"
                            "faithful"
                            "grateful"
                            "hopeful"
                            "joyful"
                            "proud"
                            "impressed"
                            "trusting"]
                "negative" ["afraid"
                            "angry"
                            "annoyed"
                            "anxious"
                            "apprehensive"
                            "ashamed"
                            "devastated"
                            "disappointed"
                            "disgusted"
                            "embarrassed"
                            "furious"
                            "guilty"
                            "jealous"
                            "lonely"
                            "sad"
                            "sentimental"
                            "terrified"]
                "neutral" ["anticipating"
                           "nostalgic"
                           "prepared"
                           "surprised"]}
        emotion-to-sentiment {})

  (for [[k v] (.items mapped)] 
    (for [emotion v]
      (setv (get emotion-to-sentiment emotion) k)))

  (setv sentiment-id (dfor [i x] (enumerate (set (.values emotion-to-sentiment))) [x i]))

  (defn ^dict sentiment->id [sample]
    (setv labels (lfor x 
                       (:context sample)
                       (-> (get sentiment-id (get emotion-to-sentiment x))
                           (torch.tensor :dtype torch.long)
                           (.unsqueeze 0))))
    {"labels" labels})

  (defn ^dict tokenize [sample]
    (tokenizer (:utterance sample) :truncation True))

  ;; batched need to be set false otherwise we cannot do the dict lookup
  (return (-> (.map empathetic-dialogues sentiment->id :batched True)
              (.map tokenize :batched True)
              (.remove-columns ["speaker_idx" "context" "utterance_idx" "utterance" "prompt" "selfeval" "tags" "conv_id"]))))


(defn atomic-relations []
  (setv turn-type-to-relation {"informative" ["capableof"
                                              "desires"
                                              "hasproperty"
                                              "madeupof"
                                              "notdesires"
                                              "objectuse"
                                              ]
                               "question" []
                               "directive" []
                               "comissive" []})

  (raise (NotImplementedError "")))

