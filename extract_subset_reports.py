import os
import shutil
from pathlib import Path

import click
import pandas as pd

BASE_PATH = Path('.')

all_reports_path = BASE_PATH / 'data' / 'wamex_xml'
subset_reports_path = BASE_PATH / 'data' / 'subset'


@click.command()
@click.option('--group', help='Group Number.')
def get_subset_reports(group):
    path = BASE_PATH / 'data' / 'report groups' / 'report_groups.csv'
    groups_df = pd.read_csv(path)

    filenames = groups_df[groups_df.group == group]['name'].values.tolist()

    if not os.path.exists(subset_reports_path):
        os.mkdir(subset_reports_path)

    for filename in filenames:
        filename = filename + '.json'
        from_path = all_reports_path / filename
        to_path = subset_reports_path / filename
        shutil.copy(str(from_path), str(to_path))


if __name__ == '__main__':
    get_subset_reports()
