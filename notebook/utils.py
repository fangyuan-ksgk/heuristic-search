# Constants 

label_key = "user_status" # human decision
comment_key = "user_comments" # human comment 
predict_key = "LLM_status" # LLM decision (v0)

# do not differentiate among challenges for now
yes_label_list = ["Yes", "Recommended"] # meaningful positive signal
neutral_label_list = ["KIV", "Maybe"]
no_label_list = ["No", "No ", "Reject"] # meaningful negative signal


# Util functions
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional
from collections import Counter
from nltk.util import ngrams
import nltk
import pandas as pd
import seaborn as sns

nltk.download('punkt') # set up for n-grams

def filter_data_with_label(data: dict, label_key: str, valid_label_list: list[str]) -> dict:
    return {k: v for k, v in data.items() if v[label_key] in valid_label_list}

def get_unique_value_count(data: dict, key: str, filter_key: Optional[str] = None, filter_value_list: Optional[list[str]] = None) -> dict:
    """ 
    Count occurrences of each unique value for a given key across all entries in data
    
    Args:
        data: Dictionary of entries
        key: Key to count unique values for
        
    Returns:
        Dictionary mapping unique values to their counts
    """
    value_counts = {}
    for entry in data.values():
        if key in entry:
            value = entry[key]
            if filter_key is not None and filter_key in entry:
                filter_value = entry[filter_key]
                if filter_value_list is not None and filter_value not in filter_value_list:
                    continue
            value_counts[value] = value_counts.get(value, 0) + 1
    return value_counts


def plot_word_counts(data: dict):
    comments = []
    for entry in data:
        comments.append(entry)

    # Combine all rejection comments into one string
    combined_text = ' '.join(comments)

    # Create word cloud specifically for rejection comments
    plt.figure(figsize=(10, 5))
    wordcloud_rejections = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100
    ).generate(combined_text)

    plt.imshow(wordcloud_rejections, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Rejection Comments')
    plt.show()
    

def get_ngrams(text_list, n: int, neg_filter: bool = False):
    # Combine and lowercase all texts
    combined_text = ' '.join(text_list).lower()
    
    # Tokenize into words (you might want to add more preprocessing)
    words = nltk.word_tokenize(combined_text)
    
    # Create bigrams using nltk.ngrams
    ngram_list = list(nltk.ngrams(words, n))
    
    # Filter for bigrams starting with 'no' or 'not'
    if neg_filter:
        filtered_ngrams = [bg for bg in ngram_list if bg[0] in ['no', 'not']]
    else:
        filtered_ngrams = ngram_list
    
    # Count bigrams
    ngram_counts = Counter(filtered_ngrams)
    
    return ngram_counts


def get_ngram_plots(text_list: list[str], n: int, neg_filter: bool = False):
    
    ngram_counts = get_ngrams(text_list, n, neg_filter)

    # Convert to dataframe for plotting
    ngram_df = pd.DataFrame([
        {'ngram': ' '.join(k), 'count': v} 
        for k, v in ngram_counts.most_common(15)
    ])

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ngram_df, x='count', y='ngram')
    if neg_filter:
        plt.title(f'Top 15 Negative {n}-grams in Rejection Comments')
    else:
        plt.title(f'Top 15 Positive {n}-grams in Acceptance Comments')
    plt.xlabel('Count')
    plt.ylabel('N-gram')
    plt.tight_layout()
    plt.show()
    
    
def get_dict_entry(entry: dict, key: str) -> str:
    if key in entry:
        return entry[key]
    elif key in entry["form_fields"]:
        return entry["form_fields"][key]
    else:
        return "None"


def convert_entry_to_prompt(grant_entry: dict) -> str:
    # Generating the prompt with the data
    prompt = f"""
    Given the following details of a grant application, determine the likelihood of acceptance. 
    Assess how well the project aligns with the challenge theme, its viability, its potential impact, and any competitive advantage. 
    Consider the stage of development, the proof of concept status, and existing partnerships. 
    Classify the likelihood of acceptance as 'High', 'Medium', or 'Low', and briefly explain your rationale.

    Grant Application Summary:
    - **Project Theme:** {get_dict_entry(grant_entry, 'Theme')}
    - **Category:** {get_dict_entry(grant_entry, 'Category')}
    - **Subcategory:** {get_dict_entry(grant_entry, 'Subcategory')}
    - **Project Title:** {get_dict_entry(grant_entry, 'title')}
    - **Grant Amount Sought:** {get_dict_entry(grant_entry, 'Amount of grant funding sought (in SGD)')}
    - **Objectives:** {get_dict_entry(grant_entry, 'Project objectives')}
    - **Have you obtained proof of concept for your project?** {get_dict_entry(grant_entry, 'Have you obtained proof of concept for your project?')}

    Solution Summary: {get_dict_entry(grant_entry, 'LLM_summary')}

    Proposed solution: {get_dict_entry(grant_entry, 'Proposed solution')}

    Proposed project's scope of work: {get_dict_entry(grant_entry, "Proposed project's scope of work")}

    Classify the likelihood of acceptance and briefly explain why this project meets or does not meet the challenge's acceptance criteria.
    """

    return prompt


def make_training_example(entry: dict) -> tuple[str, str]:
    prompt = convert_entry_to_prompt(entry)
    if entry[label_key] in yes_label_list:
        return prompt, "Yes"
    elif entry[label_key] in no_label_list:
        return prompt, "No"
    else:
        return None