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
from typing import Callable
from nltk.util import ngrams
import nltk
import pandas as pd
import seaborn as sns

nltk.download('punkt') # set up for n-grams
INVALID_COMMENTS = ["hello world", "dsd", "", "Testing if leaving the comments work", "nan"]


def fix_label(label: str) -> str:
    if label == "No ":
        return "No"
    else:
        return label
    
def fix_comment(comment: str) -> str: 
    if comment in INVALID_COMMENTS:
        return ""
    else:
        return comment

def filter_data_with_label(data: dict, label_key: str, valid_label_list: list[str]) -> dict:
    for k, v in data.items():
        v[label_key] = fix_label(v[label_key])
        v[comment_key] = fix_comment(v[comment_key])
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


def plot_pie_chart(labeled_ratio, label_counts):
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create a figure with two subplots side by side
    plt.figure(figsize=(20, 8))

    # First subplot - Labeled vs Unlabeled ratio
    plt.subplot(1, 2, 1)
    plt.pie([labeled_ratio, 1-labeled_ratio], 
            labels=['Labeled', 'Unlabeled'],
            autopct='%1.1f%%',
            colors=sns.color_palette("Set3")[0:2],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 24, 'fontweight': 'bold'},
            shadow=True)

    # Second subplot - Label distribution
    plt.subplot(1, 2, 2)
    plt.pie(label_counts.values(), 
            labels=label_counts.keys(),
            autopct='%1.1f%%',
            colors=sns.color_palette("Set3"),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 24, 'fontweight': 'bold'},
            shadow=True)

    # Add a clean look with a slight zoom
    plt.tight_layout()
    plt.show()


def plot_word_counts(data: dict):
    comments = []
    for entry in data:
        if not isinstance(entry, str):
            continue
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
    
    
def plot_phrases(data: list[str], title_str: str = "Phrase Cloud of Comments"):
    # Filter out non-string entries and create frequency dict
    phrase_freq = {}
    for phrase in data:
        if isinstance(phrase, str):
            phrase = phrase.strip()
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1

    if not phrase_freq:
        return

    # Create word cloud using the frequency dictionary directly
    plt.figure(figsize=(10, 5))
    wordcloud_phrases = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        collocations=False,  # Disable internal collocation detection
        prefer_horizontal=0.7
    ).generate_from_frequencies(phrase_freq)

    plt.imshow(wordcloud_phrases, interpolation='bilinear')
    plt.axis('off')
    plt.title(title_str)
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


def get_ngram_plots(text_list: list[str], n: int, neg_filter: bool = False, top_k: int = 15):
    
    ngram_counts = get_ngrams(text_list, n, neg_filter)

    # Convert to dataframe for plotting
    ngram_df = pd.DataFrame([
        {'ngram': ' '.join(k), 'count': v} 
        for k, v in ngram_counts.most_common(top_k)
    ])

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=ngram_df, x='count', y='ngram')
    
    if neg_filter:
        plt.title(f'Top {top_k} Negative {n}-grams in Rejection Comments')
    else:
        plt.title(f'Top {top_k} Positive {n}-grams in Acceptance Comments')
    plt.xlabel('Count')
    plt.ylabel('N-gram')
    plt.tight_layout()
    plt.show()
    
    
POS_COMMENT_KEYWORD_PROMPT = "Extract no more than 3 key words/phrases for the acceptance from the comments. Comments: {comments}. Respond in JSON format, for instance: ```json ['word1', 'phrase2', ...]``` Example: {example}"
NEG_COMMENT_KEYWORD_PROMPT = "Extract no more than 3 key words/phrases for the rejection from the comments. Comments: {comments}. Respond in JSON format, for instance: ```json ['word1', 'phrase2', ...]``` Example: {example}"

NEG_EXAMPLE = "Comment: This project is not disruptive. Response: ```json ['not disruptive']```"
POS_EXAMPLE = "Comment: Developed a new method that reduces production cost of activated carbon by up to 60%. Response: ```json ['production cost', 'activated carbon', '60% reduction']```"

BAD_KEYWORDS = ["no", "not", "Not", "No"]


def get_keyword_prompt(comment: str, neg_example: str, pos_example: str, is_neg: bool):
    if is_neg:
        return NEG_COMMENT_KEYWORD_PROMPT.replace("{comments}", comment).replace("{example}", neg_example)
    else:
        return POS_COMMENT_KEYWORD_PROMPT.replace("{comments}", comment).replace("{example}", pos_example)
    
# Keyword extraction gadget out-performs N-Gram by large margin



def keyword_extraction_with_llm(comment_counts: dict[str, int], neg_comment: bool, get_response: Callable, extract_json_from_text: Callable):
    
    extract_prompts = []
    for comment, count in comment_counts.items():
        if comment in INVALID_COMMENTS:
            continue
        prompt = get_keyword_prompt(comment, NEG_EXAMPLE, POS_EXAMPLE, neg_comment)
        extract_prompts.append(prompt)
        
    responses = get_response(extract_prompts, system_prompt="Analyze the comments.")
    
        
    extracted_keywords = []
    for response in responses:
        try: 
            keywords = extract_json_from_text(response)
            for keyword in keywords:
                if keyword in BAD_KEYWORDS:
                    continue
                extracted_keywords.append(keyword)
        except:
            continue 
        
    return extracted_keywords


def calculate_metrics(data, predict_key, label_key):
    true_no = 0    # Actually No
    pred_no = 0    # Predicted No
    correct_no = 0 # Correctly predicted No
    
    for key, entry in data.items():
        true_label = entry[label_key]
        pred_label = entry[predict_key]
        
        # Convert labels to standardized format
        true_is_no = true_label in no_label_list
        pred_is_no = pred_label in no_label_list
        
        if true_is_no:
            true_no += 1
        if pred_is_no:
            pred_no += 1
        if true_is_no and pred_is_no:
            correct_no += 1
    
    # Calculate metrics
    recall = correct_no / true_no if true_no > 0 else 0    # Correct No / Total actual No
    precision = correct_no / pred_no if pred_no > 0 else 0  # Correct No / Total predicted No
    false_discovery_rate = 1 - precision  # Incorrect No / Total predicted No
    
    return {
        'recall': recall,
        'precision': precision,
        'false_discovery_rate': false_discovery_rate,  # Added this metric
        'true_no': true_no,
        'pred_no': pred_no,
        'correct_no': correct_no
    }
    
    
def plot_metrics(metric_dict):
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with higher DPI for sharper rendering
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Prepare data
    metrics = ['Recall', 'Precision', 'False Discovery Rate']
    values = [
        metric_dict['recall'],
        metric_dict['precision'],
        metric_dict['false_discovery_rate']
    ]
    
    # Custom colors with gradient
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Create bars with alpha for transparency
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, width=0.6)
    
    # Customize title and layout
    plt.title('Classification Metrics for "No" Class', pad=20, fontsize=20, fontweight='bold')
    plt.ylim(0, 1.1)  # Increased ylim to make room for percentage labels
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}',
                ha='center', va='bottom',
                fontsize=20,
                fontweight='bold')
    
    # Add count annotations inside the bars
    counts_text = [
        f'Correct: {metric_dict["correct_no"]}\nTotal: {metric_dict["true_no"]}',
        f'Correct: {metric_dict["correct_no"]}\nPred: {metric_dict["pred_no"]}',
        f'Wrong: {metric_dict["pred_no"] - metric_dict["correct_no"]}\nPred: {metric_dict["pred_no"]}'
    ]
    
    for bar, text in zip(bars, counts_text):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                text,
                ha='center', va='center',
                color='white',
                fontsize=16,
                fontweight='bold')
    
    # Customize spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle shadow effect to bars
    for bar in bars:
        bar.set_edgecolor('none')
        
    plt.tight_layout()
    return plt
    
    
def get_dict_entry(entry: dict, key: str) -> str:
    if key in entry:
        return entry[key]
    elif key in entry["form_fields"]:
        return entry["form_fields"][key]
    else:
        return "None"


PREFIX_PROMPT = """Given the following details of a grant application, make a decision (Yes, No, Maybe) on whether to accept it or not, and a brief comment explanating your decision. 
    Assess how well the project aligns with the challenge theme, its viability, its potential impact, and any competitive advantage. 
    Consider the stage of development, the proof of concept status, and existing partnerships.
    """
    
SUFFIX_PROMPT = """Please provide your response in the following JSON format:
    ```json
    {{
        "decision": "Yes/No/Maybe",
        "comment": "Your detailed assessment explaining the decision"
    }}
    ```"""

def convert_entry_to_prompt(grant_entry: dict, use_prefix_n_suffix: bool = True) -> str:
    """ 
    This is quite important, we want diversity out of this one, too. 
    """
    
    base_prompt = f"""
    Grant Application Summary:

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
    """
    
    if use_prefix_n_suffix:
        prompt = f"""{PREFIX_PROMPT}
        {base_prompt}
        {SUFFIX_PROMPT}
        """
    else:
        prompt = base_prompt
    return prompt


def make_training_example(entry: dict, use_prefix_n_suffix: bool = True) -> tuple[str, str]:
    prompt = convert_entry_to_prompt(entry, use_prefix_n_suffix)
    assert label_key in entry, "Label not found in data dictionary"
    label = "Yes" if entry[label_key] in yes_label_list else "No"
    assert comment_key in entry, "Comment not found in data dictionary"
    comment = entry[comment_key]
    return prompt, label, comment



def process_data(data: dict, save_path: str = "../data/processed_data.json", use_prefix_n_suffix: bool = True):
    processed_data = {"prompt": [], "label": [], "comment": []}
    for i, d in enumerate(data.values()): # comment is probably more important than the decision
        te = make_training_example(d, use_prefix_n_suffix)
        if te is not None:
            processed_data["prompt"].append(te[0].strip())
            processed_data["label"].append(te[1])
            processed_data["comment"].append(te[2].strip())      
            
    import json
    with open(save_path, 'w') as f:
        json.dump(processed_data, f)
        
        
