import tomllib


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/preprocessing/cluster_graph.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def compute_krackhardt(internal, external):
    """
    Compute the Krackhardt clustering coefficient for the given internal and external edges.
    
    Parameters:
    - internal: int, the number of internal edges.
    - external: int, the number of external edges.
    
    Returns:
    - krackhardt: float, the Krackhardt clustering coefficient.
    """
    return (external - internal) / (external + internal)


def node_analysis(graph_df, cluster_of_interest):
    """
    Perform node analysis on a given cluster in the article citation graph data.
    Compute the Krackhardt E/I density for the given cluster.
    Also return the most cited and most citing external clusters.
    
    Parameters:
    - graph_df: pd.DataFrame, the graph data.
    
    Returns:
    - krackhardt: dict, the Krackhardt clustering coefficient for references and citations.
    - frequent_clusters: dict, the most cited and most citing external clusters.
    """

    cluster_subgraph = graph_df.loc[graph_df['Cluster ID'] ==
                                    cluster_of_interest]
    remaining_subgraph = graph_df.loc[graph_df['Cluster ID'] !=
                                      cluster_of_interest]

    krackhardt = {'References': 0, 'Citations': 0}
    edge_type = {'External': 0, 'Internal': 0}
    totals = {'References': edge_type.copy(), 'Citations': edge_type.copy()}
    frequent_clusters = {'References': [], 'Citations': []}

    for pmid in cluster_subgraph['Pmid']:
        # references (out-links)
        references = cluster_subgraph.loc[cluster_subgraph['Pmid'] == pmid,
                                          'Out_links'].values[0]
        external_references = remaining_subgraph.loc[
            remaining_subgraph['Pmid'].isin(references)]
        internal_references = cluster_subgraph.loc[
            cluster_subgraph['Pmid'].isin(references)]
        frequent_clusters['References'].extend(
            external_references['Cluster ID'].values)
        totals['References']['External'] += len(external_references)
        totals['References']['Internal'] += len(internal_references)

        # citations (in-links)
        citations = cluster_subgraph.loc[cluster_subgraph['Pmid'] == pmid,
                                         'In_links'].values[0]
        external_citations = remaining_subgraph.loc[
            remaining_subgraph['Pmid'].isin(citations)]
        internal_citations = cluster_subgraph.loc[
            cluster_subgraph['Pmid'].isin(citations)]
        frequent_clusters['Citations'].extend(
            external_citations['Cluster ID'].values)
        totals['Citations']['External'] += len(external_citations)
        totals['Citations']['Internal'] += len(internal_citations)

    # compute Krackhardt clustering coefficient
    krackhardt['References'] = compute_krackhardt(
        totals['References']['Internal'], totals['References']['External'])

    krackhardt['Citations'] = compute_krackhardt(
        totals['Citations']['Internal'], totals['Citations']['External'])

    most_frequent_reference_cluster = max(
        set(frequent_clusters['References']),
        key=frequent_clusters['References'].count)
    most_frequent_citation_cluster = max(
        set(frequent_clusters['Citations']),
        key=frequent_clusters['Citations'].count)

    frequent_clusters['References'] = most_frequent_reference_cluster
    frequent_clusters['Citations'] = most_frequent_citation_cluster

    return krackhardt, frequent_clusters


def edge_analysis(graph_df, edge):
    """
    Perform edge analysis on a given edge in the article citation graph data.
    Obtain the the density (weight) of references for the given edge.
    
    Parameters:
    - graph_df: pd.DataFrame, the graph data.
    - edge: tuple, the edge to analyze.
    
    Returns:
    - weight: float, the density (weight) of references for the given edge.
    """

    source_cluster = edge[0]  # source cluster
    destination_cluster = edge[1]  # destination cluster
    source_subgraph = graph_df.loc[graph_df['Cluster ID'] == source_cluster]
    destination_subgraph = graph_df.loc[graph_df['Cluster ID'] ==
                                        destination_cluster]

    totals = {'Actual': 0, 'Possible': 0}

    for pmid in source_subgraph['Pmid']:
        # references (out-links)
        references = source_subgraph.loc[source_subgraph['Pmid'] == pmid,
                                         'Out_links'].values[0]
        destination_references = destination_subgraph.loc[
            destination_subgraph['Pmid'].isin(references)]
        older_articles = destination_subgraph.loc[
            destination_subgraph['Age'] > source_subgraph.loc[
                source_subgraph['Pmid'] == pmid, 'Age'].values[0]]

        totals['Actual'] += len(destination_references)
        totals['Possible'] += len(older_articles)

    # compute weights
    if totals['Possible'] == 0:
        weight = 0
    else:
        weight = totals['Actual'] / totals['Possible']

    return weight
