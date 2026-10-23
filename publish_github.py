#!/usr/bin/env python3
"""Publish this local V2 project to an existing GitHub repository.

Run with --repo-url only for initial setup. Subsequent runs need no arguments.
Never uploads ignored local .venv, SQLite prediction history, or CSV outputs.
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=check)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url', help='Existing GitHub repository HTTPS or SSH URL (first run only)')
    parser.add_argument('--message', default='Update NFL V2 dashboard and weekly data')
    args = parser.parse_args()
    if not (ROOT / '.git').exists():
        if not args.repo_url:
            raise SystemExit('Not a git checkout. First create an EMPTY GitHub repository, then run: python publish_github.py --repo-url https://github.com/USERNAME/REPO.git')
        git('init', '-b', 'main')
    remotes = git('remote').stdout.split()
    if 'origin' not in remotes:
        if not args.repo_url:
            raise SystemExit('No origin remote. Run with --repo-url https://github.com/USERNAME/REPO.git')
        git('remote', 'add', 'origin', args.repo_url)
    print('Remote:', git('remote', 'get-url', 'origin').stdout.strip())
    # Add only project source/config, notebook, and required model data; never local secrets or history DB.
    paths = ['ML_predict_2.py','ML_Predict.py','weekly_update.py','weekly_refresh.py',
             'check_artifacts.py','publish_github.py','start_jupyter.sh','start_jupyter.command',
             'NFL_V2_Local_Workbench.ipynb','README.md','requirements.txt','runtime.txt',
             '.gitignore','.github/workflows/weekly_refresh.yml',
             'artifacts/features.joblib','artifacts/win_model_lgbm.joblib',
             'artifacts/spread_model_lgbm.joblib','artifacts/team_feature_store.parquet',
             'artifacts/training_meta.json']
    missing=[p for p in paths if not (ROOT/p).exists()]
    if missing: raise SystemExit('Missing required project files: '+', '.join(missing))
    git('add', '--', *paths)
    if git('diff','--cached','--quiet',check=False).returncode == 0:
        print('No project changes to commit.')
    else:
        print('Staged files:\n'+git('diff','--cached','--stat').stdout)
        git('commit','-m',args.message)
    print('Pushing to GitHub (Git may request authentication)...')
    subprocess.run(['git','push','-u','origin','HEAD'],cwd=ROOT,check=True)
    print('GitHub push successful. On Streamlit Community Cloud select ML_predict_2.py as entrypoint.')

if __name__ == '__main__': main()
