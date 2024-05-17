from queue import PriorityQueue
import Levenshtein


class MaxPriorityQueue:
    """A class for a priority class that is a max heap
    """

    def __init__(self):
        self.pq = PriorityQueue()

    def put_list(self, items: list):
        for item in items:
            self.put(item)

    def put(self, item):
        # Ensure the item is a tuple
        if not isinstance(item, tuple):
            raise TypeError(
                "Input should be a tuple with the priority as the first element.")

        # Negate the priority to simulate max heap
        priority = -item[0]
        self.pq.put((priority,) + item[1:])

    def get(self):
        # Get the tuple and negate the priority to return the original
        priority, *data = self.pq.get()
        return (-priority, *data)

    def empty(self):
        return self.pq.empty()

    def qsize(self):
        return self.pq.qsize()

    def to_ordered_list(self):
        """Returns an ordered list of all the items with priority 

        Returns:
            [(int, item)]
        """
        temp_list = list(self.pq.queue)
        sorted_list = sorted(temp_list, key=lambda x: -x[0], reverse=True)
        return [(-priority, *data) for priority, *data in sorted_list]

    def remove_already_discovered(self, discovered: list):
        """Remove all tuples (priority, item) from the priority queue
        whose items are also in the provided list and return a new MaxPriorityQueue.
        """
        discovered_set = {item[1] for item in discovered}

        new_pq = MaxPriorityQueue()

        while not self.pq.empty():
            priority, item = self.get()
            if item not in discovered_set:
                new_pq.put((priority, item))

        return new_pq


def add_up_priorities(max_pq: MaxPriorityQueue) -> MaxPriorityQueue:
    """Adds up priorities of items with multiple entries in the MaxPriorityQueue.

    Args:
        max_pq (MaxPriorityQueue): A MaxPriorityQueue.

    Returns:
        MaxPriorityQueue: A MaxPriorityQueue with summed up priorities.
    """

    ordered_list = max_pq.to_ordered_list()

    # Use a dictionary to aggregate priorities for the same data
    priority_dict = {}
    for (priority, data) in ordered_list:
        if data in priority_dict:
            priority_dict[data] += priority
        else:
            priority_dict[data] = priority

    # Create a new MaxPriorityQueue and add the aggregated items to it
    new_pq = MaxPriorityQueue()
    for data, combined_priority in priority_dict.items():
        new_pq.put((combined_priority, data))

    return new_pq


def get_contiguous_combinations(target_query):
    """Returns all possible (contigous) combinations
        of the words in the target query (without
        skipping any words or altering the order)

    Args:
        target_query (String): The search query

    Returns:
        [String]: The phrases that might have been meant
    """
    words = target_query.lower().split()
    target_phrases = []

    # Iterating over all words
    for i in range(len(words)):
        # Generate contiguous combinations from the current word with index i
        for j in range(i + 1, len(words) + 1):
            target_phrases.append(' '.join(words[i:j]))

    return target_phrases


def find_similar_phrases(target_phrases, corpus, threshold=0):
    """Find phrases in a corpus that are similar to any of the target phrases
    based on Levenshtein distance.
    Return both the phrase and its distance.

    Args:
        target_phrases ([String]): The phrases to search for
        corpus ([String]): The corpus of phrases to search in
        threshold (int, optional): The threshold which words to return.
                                    Defaults to 0.

    Returns:
        [(String, int)]: The sorted phrases with their minimal distance
    """

    min_distances = {}

    for target_phrase in target_phrases:

        for phrase in corpus:
            distance = Levenshtein.distance(target_phrase, phrase)

            # Calculate the percentage distance
            max_distance = max(len(target_phrase), len(phrase))
            distance_percentage = 1 - (distance / max_distance)

            # Potentially update the distance percentage for the phrase.
            if phrase not in min_distances or distance_percentage > min_distances[phrase]:
                min_distances[phrase] = distance_percentage

    # Sort the phrases based on the distance percentages and add their percentages
    sorted_phrases_with_percentages = sorted(
        min_distances.items(), key=lambda x: x[1], reverse=True)

    max_phrase_length = 0
    filtered_phrases = []

    # Filtering and determining the highest phrase length in one loop
    for (phrase, perc) in sorted_phrases_with_percentages:
        if perc >= threshold:
            filtered_phrases.append((phrase, perc))

            if len(phrase) >= max_phrase_length:
                max_phrase_length = len(phrase)

    sorted_phrases_with_percentages = [
        (perc * (len(phrase) / max_phrase_length), phrase) for phrase, perc in filtered_phrases]

    return sorted_phrases_with_percentages


def replace_spaces_with_underscores(phrase):
    return phrase.replace(" ", "_")
