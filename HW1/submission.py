import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

############################################################
# Custom Types
# NOTE: You do not need to modify these.

"""
You can think of the keys of the defaultdict as representing the positions in
the sparse vector, while the values represent the elements at those positions.
Any key which is absent from the dict means that that element in the sparse
vector is absent (is zero).
Note that the type of the key used should not affect the algorithm. You can
imagine the keys to be integer indices (e.g., 0, 1, 2) in the sparse vectors,
but it should work the same way with arbitrary keys (e.g., "red", "blue", 
"green").
"""
DenseVector = List[float]
SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


############################################################
# Problem 1a
def find_alphabetically_first_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes first
    lexicographically (i.e., the word that would come first after sorting).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find min() handy here. If the input text is an empty string,
    it is acceptable to either return an empty string or throw an error.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sorted(text.split())[0]
    # END_YOUR_CODE

############################################################
# Problem 1b
def find_frequent_words(text:str, freq:int)->Set[str]:
    """
    Splits the string |text| by whitespace
    then returns a set of words that appear at a given frequency |freq|.

    For exapmle:
    >>> find_frequent_words('the quick brown fox jumps over the lazy fox',2)
    # {'the', 'fox'}
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    word_list = text.split()
    return set([word for word in word_list if word_list.count(word) == freq])
    # END_YOUR_ANSWER 

############################################################
# Problem 1c
def find_nonsingleton_words(text: str) -> Set[str]:
    """
    Split the string |text| by whitespace and return the set of words that
    occur more than once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_list = text.split()
    return set([word for word in word_list if word_list.count(word) != 1])
    # END_YOUR_CODE

############################################################
# Problem 2a
def dense_vector_dot_product(v1:DenseVector, v2:DenseVector)->float:    
    """
    Given two dense vectors |v1| and |v2|, each represented as list,
    return their dot product.
    You might find it useful to use sum(), and zip() and a list comprehension.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return sum([x1 * x2 for x1, x2 in zip(v1, v2)])
    # END_YOUR_ANSWER

############################################################
# Problem 2b
def increment_dense_vector(v1:DenseVector, scale:float, v2:DenseVector)->DenseVector:
    """
    Given two dense vectors |v1| and |v2| and float scalar value scale, return v = v1 + scale * v2.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return [x1 + scale * x2 for x1, x2 in zip(v1, v2)]
    # END_YOUR_ANSWER

############################################################
# Problem 2c
def dense_to_sparse_vector(v:DenseVector)->SparseVector:
    """
    Given a dense vector |v|, return its sparse vector form,
    represented as collection.defaultdict(float).
    
    For exapmle:
    >>> dv = [0, 0, 1, 0, 3]
    >>> dense2sparseVector(dv)
    # defaultdict(<class 'float'>, {2: 1, 4: 3})
    
    You might find it useful to use enumerate().
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return collections.defaultdict(float, {idx: val for idx, val in enumerate(v) if val != 0})
    # END_YOUR_ANSWER


############################################################
# Problem 2d

def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Given two sparse vectors (vectors where most of the elements are zeros)
    |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.

    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    Note: A sparse vector has most of its entries as 0.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sum([v1[k] * v2[k] for k in (set(v1.keys()) | set(v2.keys()))])
    # END_YOUR_CODE


############################################################
# Problem 2e

def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector,
) -> None:
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    If the scale is zero, you are allowed to modify v1 to include any
    additional keys in v2, or just not add the new keys at all.

    NOTE: Unlike `increment_dense_vector` , this function should MODIFY v1 in-place, but not return it.
    Do not modify v2 in your implementation.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v2.keys():
        v1[key] += scale * v2[key]
    # END_YOUR_CODE

############################################################
# Problem 2f

def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Return the Euclidean distance between two locations, where the locations
    are PAIRs of numbers (e.g., (3, 5)).
    NOTE: We only consider 2-dimensional Postions as the parameters of this function. 
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.hypot(loc2[0] - loc1[0], loc2[1] - loc1[1])
    # END_YOUR_CODE

############################################################
# Problem 3a

def mutate_sentences(sentence: str) -> List[str]:
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be "similar" to the original sentence if
      - it has the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the
        original sentence (the words within each pair should appear in the same
        order in the output sentence as they did in the original sentence).
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more
        than once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse',
                 'the cat and the cat', 'cat and the cat and']
                (Reordered versions of this list are allowed.)
    """
    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    adjacent_to = collections.defaultdict(list)
    result_set = set()
    for front, back in zip(sentence.split()[:-1], sentence.split()[1:]):
        adjacent_to[front].append(back)

    def traverse_with(crafted: List[str], word_left: int):
        if word_left == 0:
            return result_set.add(" ".join(crafted))

        if (next_words := adjacent_to.get(crafted[-1])) != None:
            return [traverse_with(crafted + [next_word], word_left - 1) for next_word in next_words]

    for start in adjacent_to.keys():
        traverse_with([start], len(sentence.split()) - 1)
    
    return list(result_set)
    # END_YOUR_CODE