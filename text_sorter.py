import os
import re
import string
import functools
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
import nltk
from nltk.tokenize import PunktTokenizer
from nltk.data import find
from nltk import download

T = TypeVar('T')

class SortStrategy(Generic[T], ABC):
    """
    Strategy class that all different types of comparisons can implement
    """
    @abstractmethod
    def compare(self, item1: T, item2: T) -> int:
        """
        Comparing two items:
        -1 if item1 < item2
        0 if item1 == item2
        1 if item1 > item2
        """
        pass

class CompositeSorter(Generic[T]):
    """
    Chains together different strategies, if a tie occurs in the comparison,
    the next strategy is used.
    """

    def __init__(self):
        self.strategies: List[SortStrategy[T]] = []

    def add_strategy(self, strategy: SortStrategy[T]) -> 'CompositeSorter[T]':
        self.strategies.append(strategy)
        return self
    
    def compare(self, item1: T, item2: T) -> int:
        for strategy in self.strategies:
            result = strategy.compare(item1, item2)
            if result != 0:
                return result
        return 0

    def sort(self, items: List[T]) -> List[T]:
        return sorted(items, key=functools.cmp_to_key(self.compare))

# String comparison strategies
class StringCompareMode(ABC):
    @abstractmethod
    def compare(self, str1: str, str2: str) -> int:
        """
        Comparing two strings:
        -1 if str1 < str2
        0 if str1 == str2
        1 if str1 > str2
        """
        pass

class CasePriorityMode(StringCompareMode):
    # Sorts letters together by case sensitivity, priority given depending on lowercase_first value
    def __init__(self, lowercase_first: bool = True):
        self.lowercase_first = lowercase_first
    
    def compare(self, str1: str, str2: str) -> int:
        low_str1, low_str2 = str1.lower(), str2.lower()

        if low_str1 != low_str2:
            return -1 if low_str1 < low_str2 else 1
        
        for c1, c2 in zip(str1, str2):
            if c1.lower() == c2.lower():
                c1_is_lower = c1.islower()
                c2_is_lower = c2.islower()
                
                if c1_is_lower != c2_is_lower:
                    if self.lowercase_first:
                        return -1 if c1_is_lower else 1
                    else:
                        return 1 if c1_is_lower else -1
            else:
                return -1 if c1.lower() < c2.lower() else 1
        
        return 0

class CaseInsensitiveMode(StringCompareMode):
    def compare(self, str1: str, str2: str) -> int:
        low_str1, low_str2 = str1.lower(), str2.lower()
        return -1 if low_str1 < low_str2 else 1 if low_str1 > low_str2 else 0

class CaseSensitiveMode(StringCompareMode):
    def compare(self, str1: str, str2: str) -> int:
        return -1 if str1 < str2 else 1 if str1 > str2 else 0

class LettersOnlyMode(StringCompareMode):
    def compare(self, str1: str, str2: str) -> int:
        letters1 = ''.join(c for c in str1 if c.isalpha())
        letters2 = ''.join(c for c in str2 if c.isalpha())
        
        return -1 if letters1 < letters2 else 1 if letters1 > letters2 else 0

class AlphabeticStrategy(SortStrategy[str]):
    """
    Strategy for comparing strings using interchangeable comparison modes
    """
    # Standard modes
    class StandardModes:
        CASE_PRIORITY = CasePriorityMode()
        CASE_INSENSITIVE = CaseInsensitiveMode()
        CASE_SENSITIVE = CaseSensitiveMode()
        LETTERS_ONLY = LettersOnlyMode()
    
    def __init__(self, compare_mode: StringCompareMode, reverse: bool = False, ignore_quotes: bool = True):
        self.compare_mode = compare_mode
        self.reverse = reverse
        self.ignore_quotes = ignore_quotes
    
    def compare(self, str1: str, str2: str) -> int:
        if self.ignore_quotes:
            str1 = re.sub(r'[\'"]', '', str1)
            str2 = re.sub(r'[\'"]', '', str2)

        result = self.compare_mode.compare(str1, str2)
        return -result if self.reverse else result

class SentenceTokenizer:
    def __init__(self, source_text_path=None):
        if source_text_path != None:
            self.text = self._load_text_file(source_text_path)
        self._check_nltk_deps()

    def _load_text_file(self, source_text_path):
        try:
            f = open(source_text_path)
        except FileNotFoundError:
            print(f"File {source_text_path} does not exist")
        except PermissionError:
            print(f"No permission to access {source_text_path}")
        except IOError as e:
            print(f"An I/O error occurred: {e}")
        
        text = f.read()
        if not isinstance(text, str):
            raise InvalidDataTypeError("Data must be a string")
        f.close()

        return text

    def set_text(self, text):
        self.text = text

    def _check_nltk_deps(self):
        # Check if required dependencies for nltk and the punkt sentence tokenizer are downloaded
        try:
            find('tokenizers/punkt')
        except LookupError:
            download('punkt')

        try:
            find('tokenizers/punkt_tab')
        except LookupError:
            download('punkt_tab')

    def tokenize(self) -> list[string]:
        sent_detector = PunktTokenizer()
        sent_tokens = []

        # Remove dash dividers 
        clean_text = re.sub(r"^-{10,}\s*\n*", "", self.text, flags=re.MULTILINE)

        sent_tokens = sent_detector.tokenize(clean_text)
        return sent_tokens 

if __name__ == "__main__":
    source_text_path = "ShortStory.txt"
    sent_tokenizer = SentenceTokenizer(source_text_path)
    raw_sents = sent_tokenizer.tokenize()

    custom_sorter = CompositeSorter[str]()

    custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.CASE_PRIORITY))

    """
    Below are examples of different strategy uses for customizing the alphabetical sorting properties, they can be chained together to break ties 
    """

    # Lowercase letters are grouped under their capital counter parts 
    #custom_sorter.add_strategy(AlphabeticStrategy(CasePriorityMode(lowercase_first=False))) 

    # Case priority groups same letters together, quotes are grouped together
    #custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.CASE_PRIORITY, ignore_quotes=False)) 

    # Case priority, reverse sorting order, quotes are grouped together
    #custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.CASE_PRIORITY, reverse=True, ignore_quotes=False))

    #custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.LETTERS_ONLY))
    #custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.CASE_INSENSITIVE))
    #custom_sorter.add_strategy(AlphabeticStrategy(AlphabeticStrategy.StandardModes.CASE_SENSITIVE))

    sorted_sents = custom_sorter.sort(raw_sents)

    f = open("SortedText.txt", "w")

    for sent in sorted_sents:
        f.write("".join(sent) + "\n")
        print(sent)

    f.close()
