#!/bin/bash
cd DeepMindCheck
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn deepmindcheck.wsgi:application
