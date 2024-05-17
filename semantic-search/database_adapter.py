import sqlite3


def get_title_of_publication_with_id(publication_id):
    conn = sqlite3.connect('../ontology_creation/databases/publications-database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT title FROM Publication WHERE publication_id={publication_id}")
    title = cursor.fetchall()[0][0]
    conn.close()
    return title

def get_abstract_of_publication_with_id(publication_id):
    conn = sqlite3.connect('../ontology_creation/databases/publications-database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT abstract FROM Publication WHERE publication_id={publication_id}")
    abstract = cursor.fetchall()[0][0]
    conn.close()
    return abstract

def get_authors_of_publication_with_id(publication_id):
    conn = sqlite3.connect('../ontology_creation/databases/publications-database.db')
    cursor = conn.cursor()
    cursor.execute(f"""SELECT first_name, last_name
                        FROM
                        (SELECT author_id as wb_author_id
                        FROM written_by
                        WHERE publication_id={publication_id})
                        JOIN Author On wb_author_id=Author.author_id
                   """)
    authors = [first_name + ' ' + last_name for first_name, last_name in cursor.fetchall()]
    conn.close()
    return authors
