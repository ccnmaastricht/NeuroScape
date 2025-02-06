from Bio import Entrez
from habanero import counts


class ArticleMetadata:

    def fetch(self, pmid):
        """
        Gets metadata for a given PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID for a given article.

        Returns
        -------
        journal : str
            Journal of the article.
        year : str
            Year of the article.
        month : str
            Month of the article.
        title : str
            Title of the article.
        publication_type : str
            Publication type of the article.
        abstract : str 
            Abstract of the article.
        doi : str
            DOI of the article.
        citation_count : int
            Citation count of the article.
        """

        try:
            handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml')
            record = Entrez.read(handle)
        except Exception as e:
            return None

        publication_type = self._get_publication_type(record)
        if publication_type is None:
            return None

        try:
            idx = [
                i for i, s in enumerate(record['PubmedArticle'][0]
                                        ['PubmedData']['ArticleIdList'])
                if s.startswith('10.')
            ][0]
            doi = str(
                record['PubmedArticle'][0]['PubmedData']['ArticleIdList'][idx])
        except IndexError:
            doi = None

        citation_count = self._get_citation_count(doi)

        journal, year, month, title, abstract = self._get_article_details(
            record)

        if abstract is None:
            return None

        return journal, year, month, title, publication_type, abstract, doi, citation_count

    def _get_publication_type(self, record):
        """
        Gets the publication type for a given PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID for a given article.
            
        Returns
        -------
        publication_type : str
            Publication type of the article.
        """
        try:
            publication_type_list = record['PubmedArticle'][0][
                'MedlineCitation']['Article']['PublicationTypeList']
            primary_publication_type = str(publication_type_list[0])
        except (KeyError, IndexError):
            return None

        if primary_publication_type != 'Journal Article':
            return None

        publication_type = 'Research'
        if len(publication_type_list) > 1:
            types = [
                str(publication_type)
                for publication_type in publication_type_list
            ]
            if 'Review' in types:
                publication_type = 'Review'

        return publication_type

    def _get_citation_count(self, doi):
        """
        Gets the citation count for a given PubMed ID.

        Parameters
        ----------
        doi : str
            DOI for a given article.

        Returns
        -------
        citation_count : int
            Citation count of the article.
        """
        try:
            citation_count = counts.citation_count(doi)
        except Exception as e:
            return None

        return citation_count

    def _get_article_details(self, record):
        """
        Gets the article details for a given PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID for a given article.

        Returns
        -------
        journal : str
            Journal of the article.
        year : str
            Year of the article.
        month : str
            Month of the article.
        title : str
            Title of the article.
        abstract : str
            Abstract of the article.
        """

        attributes = {
            'journal': [
                'PubmedArticle', 0, 'MedlineCitation', 'Article', 'Journal',
                'Title'
            ],
            'year': [
                'PubmedArticle', 0, 'MedlineCitation', 'Article', 'Journal',
                'JournalIssue', 'PubDate', 'Year'
            ],
            'month': [
                'PubmedArticle', 0, 'MedlineCitation', 'Article', 'Journal',
                'JournalIssue', 'PubDate', 'Month'
            ],
            'title':
            ['PubmedArticle', 0, 'MedlineCitation', 'Article', 'ArticleTitle'],
            'abstract': [
                'PubmedArticle', 0, 'MedlineCitation', 'Article', 'Abstract',
                'AbstractText'
            ]
        }

        results = {}

        for attr, path in attributes.items():
            try:
                data = record
                for key in path:
                    data = data[key]
                results[attr] = ''.join(data)
            except (KeyError, TypeError):
                results[attr] = None
              
                
        

        return results['journal'], results['year'], results['month'], results[
            'title'], results['abstract']
