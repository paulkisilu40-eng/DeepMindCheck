#!/bin/bash
python DeepMindcheck/manage.py migrate --no-input
python DeepMindcheck/manage.py collectstatic --no-input --clear
gunicorn --timeout 300 --workers 1 --chdir DeepMindcheck deepmindcheck.wsgi:application