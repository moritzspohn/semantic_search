from rdflib import Graph, Namespace
from util import replace_spaces_with_underscores, get_contiguous_combinations, find_similar_phrases, add_up_priorities, MaxPriorityQueue
from database_adapter import get_title_of_publication_with_id, get_authors_of_publication_with_id


class SemanticSearchIndividualCalculations:

    def __init__(self, ontology_location='../ontology/semantic-search-ontology.rdf', urn_namespace="urn:semantic_search:"):
        """
        Initialize the SemanticSearchIndividualCalculations object.
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

    def semantic_search(self, starting_competencies: MaxPriorityQueue, max_results: int, include_direct_findings: bool, threshold: float):

        # Initialize the discovered publications and the fringe
        discovered = list()
        result = list()
        fringe = MaxPriorityQueue()

        # Determine the
        original_publications = self.get_publications_with_competencies(
            starting_competencies)
        add_up_priorities(original_publications)

        fringe.put_list(original_publications.to_ordered_list())
        discovered.extend(original_publications.to_ordered_list())

        if include_direct_findings:
            result = discovered

        # Iterate over depths of search
        for depth in range(1, max_results):

            # Check exit criterium
            if len(result) >= max_results:
                break
            if fringe.empty():
                break

            new_fringe = MaxPriorityQueue()
            new_competencies = MaxPriorityQueue()

            # Handle the fringe
            while not fringe.empty():
                current_publication = fringe.get()

                new_competencies.put_list(
                    self.get_competencies_of_publication(current_publication, threshold).to_ordered_list())
            # Add the priorities to reward multiple occurences
            new_competencies = add_up_priorities(new_competencies)

            # Handle all new competencies
            for (competency_priority, competency) in new_competencies.to_ordered_list():
                temporary_competency_Queue = MaxPriorityQueue()
                temporary_competency_Queue.put(
                    (competency_priority, competency))

                # Add new publications from the new competencies
                new_publications_for_competency = self.get_publications_with_competencies(
                    temporary_competency_Queue)

                new_fringe.put_list(
                    new_publications_for_competency.to_ordered_list())
            # Add the priorities to reward multiple occurences
            new_fringe = add_up_priorities(new_fringe)
            # Remove already discovered
            new_fringe = new_fringe.remove_already_discovered(discovered)

            # Marke the new publications to the new fringe
            discovered.extend(new_fringe.to_ordered_list())
            result.extend(new_fringe.to_ordered_list())

            # Update the fringe
            fringe = new_fringe

        if len(result) == 0:
            return "No matching Publications found"

        return result[:max_results]


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
