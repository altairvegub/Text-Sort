import text_sorter  
from text_sorter import SentenceTokenizer, AlphabeticStrategy, CompositeSorter, CasePriorityMode

custom_sorter = CompositeSorter[str]()
sent_tokenizer = SentenceTokenizer()

def test_empty_string():
    sent_tokenizer.set_text("")
    assert sent_tokenizer.tokenize() == []

def test_string_with_only_whitespace():
    sent_tokenizer.set_text("   \t  \n  ")
    assert sent_tokenizer.tokenize() == []

def test_single_sentence_with_period():
    sent_tokenizer.set_text("This is a sentence.")
    expected = ["This is a sentence."]
    assert sent_tokenizer.tokenize() == expected

def test_single_sentence_with_question_mark():
    sent_tokenizer.set_text("Is this a sentence?")
    expected = ["Is this a sentence?"]
    assert sent_tokenizer.tokenize() == expected

def test_single_sentence_with_exclamation_mark():
    sent_tokenizer.set_text("This is a sentence!")
    expected = ["This is a sentence!"]
    assert sent_tokenizer.tokenize() == expected

# Multiple Sentences 

def test_two_sentences_with_periods():
    sent_tokenizer.set_text("First sentence. Second sentence.")
    expected = ["First sentence.", "Second sentence."]
    assert sent_tokenizer.tokenize() == expected

def test_multiple_sentences_mixed_punctuation():
    sent_tokenizer.set_text("Hello world. How are you? I am fine!")
    expected = ["Hello world.", "How are you?", "I am fine!"]
    assert sent_tokenizer.tokenize() == expected

def test_sentences_with_varied_spacing():
    sent_tokenizer.set_text("Sentence one.  Sentence two.   Sentence three!")
    expected = ["Sentence one.", "Sentence two.", "Sentence three!"]
    assert sent_tokenizer.tokenize() == expected

# Check edge case where a title with a period such as Dr. may cause a sentence tokenizer to split
def test_sentence_tokenizer_title_prefixes():
    sent_tokenizer.set_text("Dr. Smith was really upset. He was expecting you earlier, at 9a.m but you never came.")
    assert sent_tokenizer.tokenize() == ["Dr. Smith was really upset.", "He was expecting you earlier, at 9a.m but you never came."]

# ----- Sorting ------

def create_sorter_with_strategy(*args, **kwargs):
    sorter = CompositeSorter()
    strategy = AlphabeticStrategy(*args, **kwargs)
    sorter.add_strategy(strategy)
    return sorter

def test_empty_list():
    sorter = create_sorter_with_strategy(AlphabeticStrategy.StandardModes.CASE_SENSITIVE)
    assert sorter.sort([]) == []

def test_single_item_list():
    sorter = create_sorter_with_strategy(AlphabeticStrategy.StandardModes.CASE_SENSITIVE)
    assert sorter.sort(["hello"]) == ["hello"]

# --- CASE_SENSITIVE Mode (Default ASCII-like) ---
def test_mode_case_sensitive():
    sorter = create_sorter_with_strategy(AlphabeticStrategy.StandardModes.CASE_SENSITIVE)
    unsorted = ["Apple", "apple", "Banana", "banana", "Cherry"]
    # Expected: Uppercase before lowercase (standard ASCII)
    expected = ["Apple", "Banana", "Cherry", "apple", "banana"]
    assert sorter.sort(unsorted) == expected

# --- CASE_INSENSITIVE Mode ---
def test_mode_case_insensitive():
    sorter = create_sorter_with_strategy(AlphabeticStrategy.StandardModes.CASE_INSENSITIVE)
    unsorted = ["Apple", "banana", "apple", "Banana", "CHERRY"]
    # Sorted as if all lowercase, original form preserved for ties based on stability
    expected = ["Apple", "apple", "banana", "Banana", "CHERRY"] # ("apple", "apple", "banana", "banana", "cherry")
    assert sorter.sort(unsorted) == expected

# --- CASE_PRIORITY Mode ---
# Default CASE_PRIORITY (lowercase before uppercase for same letter: aAbBcC)
def test_mode_case_priority_default_lowercase_first():
    sorter = create_sorter_with_strategy(AlphabeticStrategy.StandardModes.CASE_PRIORITY)
    unsorted = ["Banana", "banana", "Apple", "apple", "Cherry", "cherry"]
    expected = ["apple", "Apple", "banana", "Banana", "cherry", "Cherry"]
    assert sorter.sort(unsorted) == expected

# Explicit CASE_PRIORITY with lowercase_first=True (aAbBcC)
def test_mode_case_priority_explicit_lowercase_first_true():
    sorter = create_sorter_with_strategy(AlphabeticStrategy(CasePriorityMode(lowercase_first=True)))
    unsorted = ["Banana", "banana", "Apple", "apple"]
    expected = ["apple", "Apple", "banana", "Banana"]
    assert sorter.sort(unsorted) == expected

def test_option_reverse_case_sensitive():
    sorter = create_sorter_with_strategy(
        AlphabeticStrategy.StandardModes.CASE_SENSITIVE,
        reverse=True
    )
    unsorted = ["Cherry", "Apple", "banana"]
    expected = ["banana", "Cherry", "Apple"]
    assert sorter.sort(unsorted) == expected

def test_option_ignore_quotes_case_sensitive():
    sorter = create_sorter_with_strategy(
        AlphabeticStrategy.StandardModes.CASE_SENSITIVE,
        ignore_quotes=True
    )
    unsorted = ["'Cherry'", '"Apple"', "banana", "'apple'"]
    expected = ['"Apple"', "'Cherry'", "'apple'", "banana"]
    assert sorter.sort(unsorted) == expected

