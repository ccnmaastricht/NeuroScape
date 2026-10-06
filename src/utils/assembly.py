"""
Utilities for assembling the updated dataset (NeuroScape 2.0).

The citation density analysis reimplements src/utils/cluster_graph.py (node_analysis, edge_analysis)
with sparse matrices so that it scales to the full dataset. Given the same inputs it yields the same
Krackhardt E/I indices, most cited/citing clusters and density weights.
"""

import numpy as np
import pandas as pd
import igraph as ig
import scipy.sparse as sp

from src.utils.cluster_graph import compute_krackhardt


def compute_ages(years, months, reference_date):
    """
    Compute article ages in years relative to a fixed reference date, as in
    src/utils/filtering.py (compute_article_age) but with a fixed instead of the current date.
    A missing or unparsable month counts as January.

    Parameters:
    - years: array-like of int
    - months: array-like of str (e.g. 'Aug')
    - reference_date: str (ISO date)

    Returns:
    - ages: np.array of float
    """

    reference = pd.Timestamp(reference_date)
    month_numbers = pd.to_datetime(pd.Series(months, dtype='object'),
                                   format='%b',
                                   errors='coerce').dt.month.fillna(1)
    age_in_months = (reference.year - np.asarray(years, dtype=int)) * 12 + (
        reference.month - month_numbers.values)

    return age_in_months / 12


def flatten_links(pmids, links):
    """
    Flatten per-article link lists into (source index, target index) pairs. Links to articles
    outside the dataset and duplicate links are dropped (as with DataFrame.isin in node_analysis).

    Parameters:
    - pmids: array-like of int
    - links: list of lists of int

    Returns:
    - sources: np.array of int
    - targets: np.array of int
    """

    pmids = np.asarray(pmids, dtype=np.int64)
    lengths = np.fromiter((len(article_links) for article_links in links),
                          dtype=np.int64,
                          count=len(links))
    sources = np.repeat(np.arange(len(pmids)), lengths)
    flat = np.fromiter((link for article_links in links
                        for link in article_links),
                       dtype=np.int64,
                       count=lengths.sum())

    order = np.argsort(pmids)
    positions = np.searchsorted(pmids[order], flat)
    positions = np.clip(positions, 0, len(pmids) - 1)
    found = pmids[order][positions] == flat
    sources, targets = sources[found], order[positions[found]]

    pairs = np.unique(np.stack([sources, targets], axis=1), axis=0)

    return pairs[:, 0], pairs[:, 1]


def link_matrix(pmids, links):
    """
    Build a sparse adjacency matrix from per-article link lists. Entry (i, j) is 1 if article j
    appears in the link list of article i.

    Parameters:
    - pmids: array-like of int
    - links: list of lists of int

    Returns:
    - matrix: scipy.sparse.csr_matrix (n x n)
    """

    sources, targets = flatten_links(pmids, links)

    return sp.csr_matrix((np.ones(len(sources)), (sources, targets)),
                         shape=(len(pmids), len(pmids)))


def cluster_link_counts(matrix, cluster_ids, num_clusters):
    """
    Count links between clusters: entry (c, d) is the number of links from articles in cluster c
    to articles in cluster d.

    Parameters:
    - matrix: scipy.sparse matrix (n x n)
    - cluster_ids: np.array of int
    - num_clusters: int

    Returns:
    - counts: np.array (num_clusters x num_clusters)
    """

    membership = sp.csr_matrix(
        (np.ones(len(cluster_ids)), (np.arange(len(cluster_ids)), cluster_ids)),
        shape=(len(cluster_ids), num_clusters))

    return np.asarray((membership.T @ matrix @ membership).todense())


def most_frequent_external(counts):
    """
    For each cluster, the external cluster with the most links.

    Parameters:
    - counts: np.array (num_clusters x num_clusters)

    Returns:
    - clusters: np.array of int
    """

    external = counts.astype(float).copy()
    np.fill_diagonal(external, -1)

    return external.argmax(axis=1)


def citation_density(pmids, cluster_ids, ages, out_links, in_links,
                     num_clusters):
    """
    Citation density analysis between clusters (see scripts/graph_analysis/cluster_density.py).

    Parameters:
    - pmids: np.array of int
    - cluster_ids: np.array of int
    - ages: np.array of float
    - out_links: list of lists of int
    - in_links: list of lists of int
    - num_clusters: int

    Returns:
    - node_df: pd.DataFrame with Krackhardt indices and most cited/citing clusters per cluster
    - weights: np.array (num_clusters x num_clusters), reference density from source to destination
    """

    references = cluster_link_counts(link_matrix(pmids, out_links),
                                     cluster_ids, num_clusters)
    citations = cluster_link_counts(link_matrix(pmids, in_links), cluster_ids,
                                    num_clusters)

    internal_references = np.diag(references)
    external_references = references.sum(axis=1) - internal_references
    internal_citations = np.diag(citations)
    external_citations = citations.sum(axis=1) - internal_citations

    node_df = pd.DataFrame({
        'Cluster ID':
        np.arange(num_clusters),
        'Reference Krackhardt':
        compute_krackhardt(internal_references, external_references),
        'Citation Krackhardt':
        compute_krackhardt(internal_citations, external_citations),
        'Most Cited Cluster':
        most_frequent_external(references),
        'Most Citing Cluster':
        most_frequent_external(citations)
    })

    # Possible links: for each source article, the number of older articles in the destination
    possible = np.zeros((num_clusters, num_clusters))
    for destination in range(num_clusters):
        destination_ages = np.sort(ages[cluster_ids == destination])
        older = len(destination_ages) - np.searchsorted(
            destination_ages, ages, side='right')
        possible[:, destination] = np.bincount(cluster_ids,
                                               weights=older,
                                               minlength=num_clusters)

    weights = np.divide(references,
                        possible,
                        out=np.zeros_like(possible),
                        where=possible > 0)

    return node_df, weights


def density_graph(weights, cluster_order):
    """
    Build the cluster citation density graph as in scripts/graph_analysis/cluster_density.py.

    Parameters:
    - weights: np.array (num_clusters x num_clusters)
    - cluster_order: list of int, cluster IDs in the order of the cluster table

    Returns:
    - graph: igraph.Graph
    """

    edges = [(source, destination) for source in cluster_order
             for destination in cluster_order]
    graph = ig.Graph(edges=edges, directed=True)
    graph.es['weight'] = [weights[source, destination] for source, destination in edges]
    graph.vs['label'] = list(cluster_order)

    return graph


def article_citation_graph(pmids, cluster_ids, out_links):
    """
    Build the article citation graph (citing -> cited), with PubMed IDs and cluster IDs as
    string vertex attributes, as for the v1 article_citation.graphml.

    Parameters:
    - pmids: np.array of int
    - cluster_ids: np.array of int
    - out_links: list of lists of int

    Returns:
    - graph: igraph.Graph
    """

    sources, targets = flatten_links(pmids, out_links)
    edges = np.stack([sources, targets], axis=1).tolist()

    graph = ig.Graph(n=len(pmids), edges=edges, directed=True)
    graph.vs['name'] = [str(pmid) for pmid in pmids]
    graph.vs['cluster_id'] = [str(cluster_id) for cluster_id in cluster_ids]

    return graph


def resolve_link_candidates(records, pmids, dois):
    """
    Turn raw link candidates into links within the dataset. DOIs are matched case-insensitively.

    Parameters:
    - records: list of dict with 'Pmid', 'Cited In' and 'References'
    - pmids: array-like of int, all articles in the dataset
    - dois: array-like of str, their DOIs

    Returns:
    - in_links: dict mapping PubMed ID to list of citing PubMed IDs
    - out_links: dict mapping PubMed ID to list of cited PubMed IDs
    """

    pmid_set = set(int(pmid) for pmid in pmids)
    doi_to_pmid = {
        str(doi).lower(): int(pmid)
        for pmid, doi in zip(pmids, dois)
    }

    in_links, out_links = {}, {}
    for record in records:
        pmid = int(record['Pmid'])
        in_links[pmid] = sorted(
            set(citing for citing in record['Cited In']
                if citing in pmid_set and citing != pmid))
        cited = (doi_to_pmid.get(doi.lower())
                 for doi in (record['References'] or []))
        out_links[pmid] = sorted(
            set(cited_pmid for cited_pmid in cited
                if cited_pmid is not None and cited_pmid != pmid))

    return in_links, out_links


def symmetrize_links(in_links, out_links):
    """
    Make in-links and out-links consistent: if a cites b, then b is in a's out-links and a is in
    b's in-links (as update_links in scripts/ingestion/build_adjacencies.py).

    Parameters:
    - in_links: dict mapping PubMed ID to iterable of PubMed IDs
    - out_links: dict mapping PubMed ID to iterable of PubMed IDs

    Returns:
    - in_links, out_links: dicts mapping PubMed ID to sorted lists, covering all PubMed IDs
    """

    in_sets = {pmid: set(links) for pmid, links in in_links.items()}
    out_sets = {pmid: set(links) for pmid, links in out_links.items()}

    for pmid, cited in list(out_sets.items()):
        for cited_pmid in cited:
            in_sets.setdefault(cited_pmid, set()).add(pmid)
    for pmid, citing in list(in_sets.items()):
        for citing_pmid in citing:
            out_sets.setdefault(citing_pmid, set()).add(pmid)

    pmids = set(in_sets) | set(out_sets)

    return ({pmid: sorted(in_sets.get(pmid, ())) for pmid in pmids},
            {pmid: sorted(out_sets.get(pmid, ())) for pmid in pmids})
