# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I built and compared both models — the rule-based classifier and the ML classifier.

**Intended purpose:**  
Classify short, informal text posts (social-media style messages) into one of four mood labels: positive, negative, neutral, or mixed — so a student or small app could get a quick, cheap signal about the tone of a short message.


Files to run:

main.py — runs the rule-based demo and evaluation
ml_experiments.py — trains and evaluates the ML model
scratch.py — a debugging script used to call m.explain() on individual sentences during development

**How it works (brief):**  
Rule-based: Tokenizes text (lowercases, strips punctuation, keeps apostrophes for contractions), checks each token against fixed positive/negative word lists, applies negation handling (flips the sentiment of the word immediately following "not," "no," or "never"), applies a contrast rule that checks for opposing-sentiment words on either side of "but," checks a small set of hardcoded emoji signals against the raw (untokenized) text, and converts the resulting numeric score into a label using fixed thresholds.

ML: Converts each post into a bag-of-words vector using CountVectorizer, then trains a classifier on SAMPLE_POSTS / TRUE_LABELS to learn which word patterns correlate with which labels.



## 2. Data

**Dataset description:**  
Started with 6 example posts in dataset.py, expanded to 16 by adding realistic language: slang, sarcasm, emojis, and mixed-emotion sentences. This is a very small, curated dataset — class-exercise size, not production size.

**Labeling process:**  
Labels were assigned by hand based on how the sentence read to me. Several were genuinely contestable:

"😅😂💀" could reasonably be read as amused/positive (laughing hard at something) or as neutral (no real textual content to analyze). I labeled it neutral in the dataset, but on reflection I think positive is the more defensible read — this exact emoji combination (nervous-laugh + crying-laughing + skull) is near-universal shorthand on social media for "I'm dying laughing," not an absence of emotion. A neutral label would only make sense if the poster felt nothing, which isn't what this combination signals in practice. I'm keeping the original neutral label in the dataset itself rather than changing it retroactively, but I'm flagging this as a labeling mistake I'd fix with more time — it's a good example of how even labels I assigned confidently can be wrong on a second look, and it shows why multiple labelers matter for a dataset like this.


"The group chat was so chaotic today... but I'm just here for the drama" could be read as neutral (detached observation) or positive/mixed (enjoying the chaos) — I labeled it neutral, but a friend reading it might reasonably disagree.




**Important characteristics of your dataset:**  
- Contains slang ("no cap," "lowkey," "ghosted," "fire")
- sarcasm ("Obviously I'm the perfect person to be running on zero sleep and three coffees")
- an emoji-only post, and several posts with genuinely mixed emotional content (grief plus acceptance, exhaustion plus pride).

**Possible issues with the dataset:**  
Very small (16 examples) — nowhere near enough to represent the diversity of real informal language, and too small to meaningfully train or evaluate an ML model.

Labels reflect one person's interpretation of tone; someone from a different age group, region, or online community might label several of these differently.

Class imbalance — "mixed" and "neutral" are underrepresented relative to "positive" and "negative."

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Base positive/negative word lists, expanded during testing to add words that came up in real failures: proud, hopeful, free (positive) and ghosted, dreading, chaotic (negative).
Negation handling: flips the sentiment contribution of the word immediately following "not," "no," or "never" (verified via explain() — the token happy correctly contributed -1 in "I am not happy about this," labeled negative (negated)).
A "but" contrast rule: checks whether a positive word appears on one side of "but" and a negative word on the other, and returns mixed if so.
Emoji signals: a small hardcoded set (:), 😂, 🥲 positive; :(, 💀 negative) is checked against the raw text directly, bypassing tokenization — this matters because multi-emoji strings like "😅😂💀" don't get split into separate tokens by the current preprocessing, so token-level matching alone would miss them entirely.
Final label thresholding: score > 0 → positive, score < 0 → negative, score == 0 → neutral, with the "but" rule and a abs(score) <= 1 + mixed-signal check overriding this into "mixed" when applicable.

**Strengths of this approach:**  
Fully transparent — every prediction can be explained token-by-token via explain(), which shows exactly which words or emoji contributed and by how much.
Negation handling works reliably on the cases tested.
No training data required; behavior is fully predictable given the rules.

**Weaknesses of this approach:**  
Vocabulary-dependent, not meaning-dependent. "Got ghosted by my best friend... Not cool" originally scored exactly 0 — every single token, including "ghosted" and "cool," came back neutral in explain(), because neither word was in the lexicon. The model had zero signal despite the sentence being clearly negative to a human reader.
Cannot detect sarcasm, structurally. "Obviously I'm the perfect person to be running on zero sleep and three coffees" scored 0 — every token is neutral by the model's logic, because there are no explicitly negative words in the sentence. This is not a vocabulary gap that more words can fix; the negativity lives entirely in tone, which a keyword-based system cannot represent.
The "but" rule inherits the same vocabulary limitation. It failed on "My plant died and somehow I'm the only one who's crying about it, but... it had a good run" (predicted positive, true mixed) because "died" and "crying" were never in the negative word list, so the left side of "but" registered as having no negative signal, and the rule never fired — even though "but" was present.




## 4. How the ML Model Works (if used)

**Features used:**  
Bag-of-words representation via CountVectorizer.

**Training data:**  
Trained on the same 16 posts in SAMPLE_POSTS / TRUE_LABELS.

**Training behavior:**  
When evaluated on the same data it was trained on, the model reported 1.00 accuracy. This is not meaningful evidence of quality — it means the model had already seen and memorized every test sentence during training, which is a textbook overfitting result on a dataset this small (no held-out test set, no cross-validation).

To actually test generalization, I tried new sentences the model had never seen:


"I guess three hours of sleep and cold coffee is basically a personal best" → predicted negative
"Sure, running on fumes is exactly how I wanted to spend today" → predicted negative
"Wow, I really love waiting forty-five minutes for a bus that never came" → predicted neutral (missed the sarcasm entirely)
"Oh great, another meeting that could have been an email" → predicted negative


The first two "successes" are likely not evidence of sarcasm understanding — they closely echo the sleep-deprivation vocabulary in the training example "Obviously I'm the perfect person to be running on zero sleep and three coffees," so the model may simply have learned a correlation between exhaustion-related words and negative labels, not a general grasp of ironic tone. The bus example makes this clearer: because "love" is strongly associated with positive labels in training, the model missed the sarcasm completely and landed on neutral.



**Strengths and weaknesses:**  
Learns word-label associations automatically, without hand-written rules; in principle could pick up patterns a person wouldn't think to encode.

Weaknesses: With only 16 training examples, in-sample accuracy (1.00) is not a reliable quality measure — it reflects memorization. Generalization to new sentences is inconsistent, and apparent successes are hard to distinguish from coincidental lexical overlap with training data rather than genuine tone understanding.

## 5. Evaluation

**How you evaluated the model:**  
Ran both models against the same 16 labeled posts in dataset.py. For the ML model, additionally tested on new, unseen sentences, since the built-in evaluation only measures performance on training data.
Rule-based accuracy: 0.75 (improved from an initial 0.44 after adding missing vocabulary and the "but" contrast rule).

ML accuracy on training data: 1.00 (not a reliable generalization measure — see Section 4).


**Examples of correct predictions:**  
"I love this class so much" → positive — clear, unambiguous positive keyword.
"I am not happy about this" → negative — negation correctly flips "happy" to a negative contribution.
"Feeling tired but kind of hopeful" → mixed — after adding "hopeful" to the vocabulary, the "but" rule correctly detected opposing sentiment on either side.

**Examples of incorrect predictions:**  
"Obviously I'm the perfect person to be running on zero sleep and three coffees." — Rule-based: predicted neutral (true: negative) because no individual word is negative. ML: correctly predicted negative on this exact sentence, but this is training data the model had memorized, so it's not evidence of real sarcasm detection.
"My plant died and somehow I'm the only one who's crying about it, but... it had a good run." — Rule-based: predicted positive (true: mixed), because "died"/"crying" aren't in the negative vocabulary, so the "but" contrast never triggered.

## 6. Limitations

The dataset is very small (16 examples) — not enough to represent real-world language diversity, and too small to train or meaningfully evaluate an ML model.
The rule-based model cannot detect sarcasm or tone inversion under any circumstances. This is a structural limitation of keyword-based scoring, not a bug that more words can fix.
The rule-based model's accuracy is entirely bounded by vocabulary coverage — every fix made during this project involved adding a missing word, and there will always be more missing words for new sentences.
The ML model's apparent successes on new sentences can't be confidently attributed to real generalization versus surface-level lexical coincidence, given the tiny training set and lack of a held-out test set.
Neither model distinguishes genuine ambiguity (like "😅😂💀") from clear-cut cases — both apply the same mechanical logic regardless of how uncertain a case is, and neither can express uncertainty in its output.



## 7. Ethical Considerations

Misclassifying distress as neutral could be genuinely harmful if a system like this were used to monitor wellbeing or flag concerning messages — the rule-based model's tendency to default to "neutral" when it lacks the right vocabulary means real distress could go undetected simply because specific words weren't in its list.
Slang and dialect not represented in the training data could be systematically misread for certain age groups, regions, or online communities, since the labeling and vocabulary reflect one person's linguistic frame of reference.
Analyzing personal messages for mood raises privacy concerns independent of accuracy — even a technically "working" system reduces someone's emotional state to a label they never chose or consented to.

## 8. Ideas for Improvement

Add substantially more labeled data, ideally from multiple labelers, to reduce reliance on one person's judgment and surface disagreement on ambiguous cases.
Hold out 20–30% of the dataset as a real test set (or use cross-validation) for the ML model, rather than evaluating on training data, to get an honest generalization measure.
Try TF-IDF instead of raw word counts for the ML model, and/or limit n-gram features, to reduce memorization on small datasets.
Improve emoji tokenization so multi-emoji strings split into individually scorable tokens, and expand the emoji vocabulary beyond the current 4 hardcoded symbols.
Collect more labeled examples specifically containing sarcasm and implicit frustration, since these were the dominant failure mode for both models.