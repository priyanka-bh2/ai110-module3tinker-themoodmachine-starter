"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("My boss threw a red flag at my proposal and now I’m dreading the next meeting.")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Why does everyone act like two hours is enough time to finish this project?")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Obviously I’m the perfect person to be running on zero sleep and three coffees.")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("The group chat was so chaotic today, it’s giving main character energy but I’m just here for the drama.")
TRUE_LABELS.append("neutral")

SAMPLE_POSTS.append("I hated leaving that job, but honestly, I’m already proud of what I accomplished there.")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("😅😂💀")
TRUE_LABELS.append("neutral")

SAMPLE_POSTS.append("Got ghosted by my best friend after I literally waited an hour for them. Not cool.")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("My plant died and somehow I’m the only one who’s crying about it, but like, hey, it had a good run.")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("Finally submitted the final and my brain is officially out of service. I feel so done and so free.")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("The coffee shop was loud, the playlist was fine, and I managed to focus for, like, twenty minutes.")
TRUE_LABELS.append("neutral")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
