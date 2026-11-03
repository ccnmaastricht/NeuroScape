"""
Build proxy journal lists for years before Scimago coverage (NeuroScape 2.0, 1980-1998).

Scimago starts in 1999. For earlier years the journal set is approximated by the journals that
satisfy the v1 quartile rule (Neuroscience Q1 + Q2, Multidisciplinary Q1) in at least
`min_years` of the core years (default: 3 of 1999-2003). Renamed journals are followed back via
the NLM Catalog ('Preceding' titles, resolved by ISSN) so that their earlier titles are scraped too.

Output, one file per year in Scimago format (';'-separated), so that `scraping.py
--journal_lists scimago_proxy` can read them unchanged:
    Internal/Reference/ScimagoProxy/<discipline>/scimagojr <year> Subject Area - <discipline>.csv
plus a documentation table `proxy_journals_<discipline>.csv` in the same folder.

Columns: Title (Scimago title, or the MEDLINE title abbreviation for a preceding title), Issn,
SJR Best Quartile (modal in-rule quartile over the core years), Areas, Proxy Source
('core' or 'preceding'), Successor (the core journal a preceding title belongs to),
First Year, End Year (NLM publication years; NaN if unknown).
"""

import os
import re
import glob
import time
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from src.utils.parsing import parse_directories

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
EMAIL = os.environ['EMAIL']

EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
RULES = {'Neuroscience': ('Q1', 'Q2'), 'Multidisciplinary': ('Q1', )}


def eutils(endpoint, params, attempts=5):
    params = {**params, 'tool': 'neuroscape', 'email': EMAIL}
    for attempt in range(attempts):
        try:
            time.sleep(0.4)
            with urlopen(f'{EUTILS}/{endpoint}.fcgi?{urlencode(params)}',
                         timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10)


def format_issn(issn):
    issn = issn.strip().replace('-', '').upper()
    return f'{issn[:4]}-{issn[4:]}' if len(issn) == 8 else None


def lookup_issns(issns):
    """
    Find the NLM Catalog journal record for any of the given ISSNs.

    Returns:
    - dict with MedlineTA, FirstYear, EndYear, Preceding (list of ISSNs), or None
    """
    issns = [i for i in (format_issn(i) for i in issns) if i]
    if not issns:
        return None
    term = ' OR '.join(f'{i}[issn]' for i in issns)
    ids = re.findall(
        rb'<Id>(\d+)</Id>',
        eutils('esearch', {
            'db': 'nlmcatalog',
            'term': f'({term}) AND ncbijournals[filter]'
        }))
    if not ids:
        return None
    root = ET.fromstring(
        eutils('efetch', {
            'db': 'nlmcatalog',
            'id': ids[0].decode(),
            'retmode': 'xml'
        }))
    record = root.find('.//NLMCatalogRecord')
    if record is None:
        return None

    def year(tag):
        value = record.findtext(f'.//{tag}')
        return int(value) if value and value.isdigit() else None

    preceding = []
    for related in record.findall('TitleRelated'):
        if related.get('TitleType') == 'Preceding':
            preceding.append([i.text for i in related.findall('ISSN') if i.text])
    return {
        'MedlineTA': record.findtext('MedlineTA'),
        'FirstYear': year('PublicationFirstYear'),
        'EndYear': year('PublicationEndYear'),
        'Preceding': preceding,
    }


def core_journals(folder, discipline, core_years, min_years):
    rule = RULES[discipline]
    in_rule = {}
    for year in core_years:
        file = os.path.join(folder,
                            f'scimagojr {year} Subject Area - {discipline}.csv')
        df = pd.read_csv(file, sep=';')
        df = df[df['SJR Best Quartile'].isin(rule)]
        for title, quartile, issn, areas in df[[
                'Title', 'SJR Best Quartile', 'Issn', 'Areas'
        ]].itertuples(index=False, name=None):
            in_rule.setdefault(title, []).append((quartile, issn, areas))
    rows = []
    for title, entries in in_rule.items():
        if len(entries) < min_years:
            continue
        quartiles = Counter(e[0] for e in entries)
        quartile = min(q for q, n in quartiles.items()
                       if n == max(quartiles.values()))
        rows.append({
            'Title': title,
            'Issn': entries[-1][1],
            'SJR Best Quartile': quartile,
            'Areas': entries[-1][2],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--discipline', default='Neuroscience')
    parser.add_argument('--start_year', type=int, default=1980)
    parser.add_argument('--end_year', type=int, default=1998)
    parser.add_argument('--core_start', type=int, default=1999)
    parser.add_argument('--core_end', type=int, default=2003)
    parser.add_argument('--min_years', type=int, default=3)
    parser.add_argument('--max_depth', type=int, default=3,
                        help='How many renames to follow back per journal.')
    args = parser.parse_args()

    directories = parse_directories()
    reference = directories['internal']['reference']
    scimago_folder = os.path.join(BASEPATH, reference['scimago'],
                                  args.discipline)
    output_folder = os.path.join(BASEPATH, reference['scimago_proxy'],
                                 args.discipline)
    os.makedirs(output_folder, exist_ok=True)

    core = core_journals(scimago_folder, args.discipline,
                         range(args.core_start, args.core_end + 1),
                         args.min_years)
    print(f'{args.discipline}: {len(core)} core journals')

    rows = []
    for n, journal in enumerate(core.to_dict('records'), 1):
        quartile = journal['SJR Best Quartile']
        record = lookup_issns(str(journal['Issn']).replace('"', '').split(','))
        rows.append({
            'Title': journal['Title'],
            'Issn': journal['Issn'],
            'SJR Best Quartile': quartile,
            'Areas': journal['Areas'],
            'Proxy Source': 'core',
            'Successor': None,
            'First Year': record['FirstYear'] if record else None,
            'End Year': record['EndYear'] if record else None,
        })
        # Follow renames back while the earlier title was published within the proxy range
        frontier, seen = (record['Preceding'] if record else []), set()
        for _ in range(args.max_depth):
            next_frontier = []
            for issns in frontier:
                key = tuple(sorted(issns))
                if key in seen:
                    continue
                seen.add(key)
                previous = lookup_issns(issns)
                if previous is None or not previous['MedlineTA']:
                    continue
                if previous['EndYear'] and previous['EndYear'] < args.start_year:
                    continue
                rows.append({
                    'Title': previous['MedlineTA'],
                    'Issn': ', '.join(issns),
                    'SJR Best Quartile': quartile,
                    'Areas': journal['Areas'],
                    'Proxy Source': 'preceding',
                    'Successor': journal['Title'],
                    'First Year': previous['FirstYear'],
                    'End Year': previous['EndYear'],
                })
                next_frontier.extend(previous['Preceding'])
            frontier = next_frontier
        if n % 25 == 0:
            print(f' {n}/{len(core)} journals looked up')

    journals = pd.DataFrame(rows).drop_duplicates('Title')
    journals.to_csv(os.path.join(output_folder,
                                 f'proxy_journals_{args.discipline}.csv'),
                    index=False)
    print(f' {(journals["Proxy Source"] == "preceding").sum()} preceding titles added')

    for year in range(args.start_year, args.end_year + 1):
        active = journals[(journals['First Year'].isna()
                           | (journals['First Year'] <= year))
                          & (journals['End Year'].isna()
                             | (journals['End Year'] >= year))]
        active.to_csv(os.path.join(
            output_folder, f'scimagojr {year} Subject Area - {args.discipline}.csv'),
                      sep=';',
                      index=False)
        print(f' {year}: {len(active)} journals')


if __name__ == '__main__':
    main()
