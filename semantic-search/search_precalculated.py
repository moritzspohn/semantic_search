import numpy as np
import os
from rdflib import Graph, Namespace, URIRef, RDF
from tqdm import tqdm
from util import replace_spaces_with_underscores, get_contiguous_combinations, find_similar_phrases, add_up_priorities, MaxPriorityQueue
from database_adapter import get_title_of_publication_with_id, get_authors_of_publication_with_id


MINIMUM_SIMILARITY = 0.01

class SemanticSearchPrecalculated:
    """Semantic Search class and search algorithm for the ontology with precalculated similarities of the publications
        The class is similar to the SemancticSearch class for the individual calculations.
        The most prominent difference is the search algorithm, which can be found in the semantic search function.
    """
    def __init__(self, ontology_location='../ontology/semantic-search-ontology-precalculated-similarities.rdf', urn_namespace="urn:semantic_search:"):
        """
        Initialize the SemanticSearch object. 
        """

        self.namespace = Namespace(urn_namespace)

        all_competencies_query = f"""
            PREFIX n: <{self.namespace}>
            SELECT ?competency_name WHERE {{
                ?competency a n:Competency ;
                            n:competencies:name ?competency_name .
            }}
        """
        # Loading the graph
        self.graph = Graph()
        self.graph.parse(ontology_location, format='xml')

        # Extracting and saving the entire corpus of competencies of the ontology
        all_competencies = self.graph.query(all_competencies_query)
        self.corpus_with_underscores = [
            str(competency[0]) for competency in all_competencies]

    def get_urn_of_competency(self, competency_name):
        resulting_urn = self.namespace + competency_name
        return resulting_urn

    def search_for_query(self, search_query, input_threshold=1, search_threshold=0.5, max_results=100, include_direct_findings=True):
        """
        Search the corpus based on the search query and return the results.

        Parameters:
        - query: A string containing the user's search query.
        - threshold (optional): An integer in range (0,1) determining how accurate the matching of competencies has to be.
                        Defaults to an exact match (1).
        - max_results (optional): An integer determining how many search resulte are asked for.
                        Defaults to 100

        Returns:
        A list of results or matched documents.
        """

        assert 0 <= input_threshold <= 1

        # Determine all potential query phrases to search for in the competencies
        all_query_phrases = [replace_spaces_with_underscores(
            phrase) for phrase in get_contiguous_combinations(search_query)]

        # Determine the competencies that correlate to the search query
        query_competencies = find_similar_phrases(
            all_query_phrases, self.corpus_with_underscores, input_threshold)

        query_competencies_urns = [(priority, f"{self.namespace}Competency:{competency}") for priority, competency in query_competencies]

        query_competencies_urns_queue = MaxPriorityQueue()
        query_competencies_urns_queue.put_list(query_competencies_urns)

        return self.semantic_search(query_competencies_urns_queue, max_results, include_direct_findings, search_threshold)

    def semantic_search(self, starting_competencies: MaxPriorityQueue, max_results: int, include_direct_findings: bool, search_threshold: float):

        assert (os.path.exists('../ontology/semantic-search-ontology-precalculated-similarities.rdf'))

        # Initialize the discovered publications and the fringe
        discovered = list()
        result_publications = list()
        fringe = MaxPriorityQueue()

        # Determine the publications that contain the starting competencies
        original_publications = self.get_publications_with_competencies(
            starting_competencies, search_threshold)
        original_publications = add_up_priorities(original_publications)

        # Adding the original publications to the fringe and the discovered list
        fringe.put_list(original_publications.to_ordered_list())
        discovered.extend(original_publications.to_ordered_list())

        # If the direct findings should be included the result contains the original publications, which are currently saved in the discovered list
        if include_direct_findings:
            result_publications = discovered

        # Iterate over depths of search
        for depth in range(1, max_results):

            # Check exit criterium
            if len(result_publications) >= max_results:
                break
            if fringe.empty():
                break

            new_publications = MaxPriorityQueue()

            # Handle all publications from the fringe
            while not fringe.empty():
                current_publication = fringe.get()
                # Add all similar publications to the current publication to the new publications priority queue
                new_publications.put_list(
                    self.get_publications_similar_to_publication(current_publication, search_threshold).remove_already_discovered(discovered).to_ordered_list())

            # Add the priorities to reward multiple occurences
            new_publications = add_up_priorities(new_publications)

            # Add the new publications to the discovered and the results lists
            discovered.extend(new_publications.to_ordered_list())
            result_publications.extend(new_publications.to_ordered_list())

            # Setup the fringe for the next depth iteration
            fringe = new_publications

        if len(result_publications) == 0:
            return "No matching Publications found"

        # Return the resulting publications in a readable form
        return result_publications[:max_results]


    def get_publications_with_competencies(self, competencies: MaxPriorityQueue, threshold=0):
        publications_with_competencies = MaxPriorityQueue()
        for competency_priority, competency in competencies.to_ordered_list():

            competency_urn = competency

            publications_query = f"""
                PREFIX n: <{self.namespace}>
                SELECT ?derived_from_relation ?publication WHERE {{
                <{competency_urn}> ?derived_from_relation ?publication .
                FILTER(STRSTARTS(STR(?derived_from_relation), STR(n:Extraction)))
                ?publication a n:Publication .
                    }}
                """

            publications_with_certainties = list(self.graph.query(publications_query))

            # Iterating over the publications to add them to the found publications
            for certainty_uri_ref, publication in publications_with_certainties:
                publication_certainty = self.get_certainty_from_URI_Ref(certainty_uri_ref)

                new_publication_priority = competency_priority * publication_certainty

                # Only return publications with a priority at least as high as the threshold
                if publication_certainty >= threshold:
                    publications_with_competencies.put((new_publication_priority, str(publication)))

        return publications_with_competencies


    def get_certainty_from_URI_Ref(self, uri_ref):
        parts = str(uri_ref).split('-')
        return float(parts[-1])

    def get_competencies_of_publication(self, publication_tuple, threshold=0):

        competencies_of_publication = MaxPriorityQueue()

        publication_priority, publication = publication_tuple

        publications_query = f"""
            PREFIX n: <{self.namespace}>
            SELECT ?derived_from_relation ?competency WHERE {{
            ?competency ?derived_from_relation <{publication}> .
            FILTER(STRSTARTS(STR(?derived_from_relation), STR(n:Extraction)))
            ?competency a n:Competency .
                }}
            """

        competencies_with_certainties = list(self.graph.query(publications_query))
        # Iterating over the publications to add them to the found publications
        for certainty_uri_ref, competency in competencies_with_certainties:
            competency_certainty = self.get_certainty_from_URI_Ref(certainty_uri_ref)

            # Only return competencies with a priority at least as high as the threshold
            if competency_certainty < threshold:
                continue

            new_competency_priority = publication_priority * competency_certainty
            competencies_of_publication.put((new_competency_priority, str(competency)))

        return competencies_of_publication

    def make_publications_readable(self, publication_list):
        output_string = ""
        for certainty, publication in publication_list:

            publication_id = self.get_publication_id_from_urn(publication)
            publication_title = get_title_of_publication_with_id(publication_id)
            authors = get_authors_of_publication_with_id(publication_id)
            rounded_certainty = round(certainty, 4)
            output_string = output_string + f"Title of Publication {publication_id} with certainty {rounded_certainty}: \n{publication_title}\n"
            for author in authors:
                output_string = output_string + author + "\n"
            output_string = output_string + "\n"

        return output_string

    def get_publication_id_from_urn(self, publication_urn):
        return publication_urn.split(':')[-1]


    def get_all_publications(self):
        all_publications_query = f"""
                PREFIX n: <{self.namespace}>
                SELECT ?publication WHERE {{
                ?publication a n:Publication .
                    }}
                """
        # Execute the query
        results = self.graph.query(all_publications_query)

        # Convert the results to a list and return
        publications = [str(result[0]) for result in results]
        return publications

    def add_all_publication_pairs_to_ontology(self, all_publications_competencies_list_dict: dict):
        all_publications = list(all_publications_competencies_list_dict.keys())

        similarities = np.zeros((len(all_publications), len(all_publications)))
        # Fill the diagonal with NaN
        np.fill_diagonal(similarities, np.nan)

        # Calculate all similarities of all publication pairs, the order is irrelevant, because the similarities are symmetric
        for i in tqdm(range(0, len(all_publications))):
            current_publication = all_publications[i]
            current_publication_competencies = all_publications_competencies_list_dict[current_publication]
            for j in range(i + 1, len(all_publications)):
                similar_publication = all_publications[j] 
                similar_publication_competencies = all_publications_competencies_list_dict[similar_publication]
                # Calculate the similarities
                similarity = self.get_similarity_of_competency_lists(current_publication_competencies, similar_publication_competencies)
                # Save the similarities
                similarities[i][j] = similarity
                similarities[j][i] = similarity

        # Compute min and max for each row excluding NaN values
        similarities_min = np.nanmin(similarities)
        similarities_max = np.nanmax(similarities)

        # Scale each row so that all entries fall into [0,1]
        scaled_similarities = (similarities - similarities_min) / (similarities_max - similarities_min)

        # Add the scaled similarities to the ontology graph
        for i in tqdm(range(0, len(all_publications))):
            for j in range(i + 1, len(all_publications)):
                if scaled_similarities[i][j] >= MINIMUM_SIMILARITY:
                    self.add_publication_similarity_to_graph(all_publications[i], all_publications[j] , scaled_similarities[i][j])

        # Save the graph to an XML file
        self.graph.serialize(destination='../ontology/semantic-search-ontology-precalculated-similarities.rdf', format='xml')


    def get_similarity_of_competency_lists(self, competency_list_1, competency_list_2):
        # Calculate the sompetency similarities of two lists that contain (priority, competency) tuples
        # If both lists contain a competency the similarity is the product of the priorities
        similarity_scores = [(priority1 * priority2, competency1)
              for priority1, competency1 in competency_list_1
              for priority2, competency2 in competency_list_2
              if competency1 == competency2]
        # The total similarity is the sum of all individual competency similarities
        total_priority = sum([priority for priority, _ in similarity_scores])
        return total_priority

    def add_publication_similarity_to_graph(self, urn_publication_1, urn_publication_2, similarity: float):
        # function to add a derived_from relation

        assert 0 <= similarity <= 1

        new_dynamic_similarity = URIRef(self.namespace + f"Similarity:similarity-{similarity}")

        class_publication = URIRef(self.namespace + "Publication")
        uri_ref_publication_1 = URIRef(urn_publication_1)
        uri_ref_publication_2 = URIRef(urn_publication_2)

        # Ensure that both the publications exist in the graph before creating the relationship
        if (uri_ref_publication_1, RDF.type, class_publication) in self.graph and (uri_ref_publication_2, RDF.type, class_publication) in self.graph:

            # Add new similarity to the graph
            self.graph.add((uri_ref_publication_1, new_dynamic_similarity, uri_ref_publication_2))
            return new_dynamic_similarity
        else:
            raise IOError("At least one of the publications does not exist")

    def get_publications_similar_to_publication(self, publication_tuple, search_threshold=0):

        publication_priority, publication_urn = publication_tuple

        similar_publications = MaxPriorityQueue()

        # The query is indifferent to the order of the publicaitons
        similar_publications_query = f"""
            PREFIX n: <{self.namespace}>
            SELECT ?similarity_relation ?similar_publication WHERE {{
                    {{
                        <{publication_urn}> ?similarity_relation ?similar_publication .
                        FILTER(STRSTARTS(STR(?similarity_relation), STR(n:Similarity)))
                        ?similar_publication a n:Publication .
                    }}
                    UNION
                    {{
                        ?similar_publication ?similarity_relation <{publication_urn}> .
                        FILTER(STRSTARTS(STR(?similarity_relation), STR(n:Similarity)))
                        ?similar_publication a n:Publication .
                    }}
                }}
            """

        publications_with_similarities = list(self.graph.query(similar_publications_query))

        for similarity_relation, current_similar_publication in publications_with_similarities:
            current_publication_similarity = float(str(similarity_relation).split('-')[-1])

            # Only return publications with a priority at least as high as the threshold
            if current_publication_similarity < search_threshold:
                continue

            # Calculate the new priority that is the product of the old priority times the similarity of the publications
            new_publication_priority = publication_priority * current_publication_similarity
            similar_publications.put((new_publication_priority, str(current_similar_publication)))

        return similar_publications
