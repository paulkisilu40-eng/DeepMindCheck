#!/bin/bash
# Check if we're already in the right directory
if [ -f "manage.py" ]; then
    echo "Found manage.py in current directory"
    python manage.py collectstatic --noinput
    python manage.py migrate
    gunicorn deepmindcheck.wsgi:application
elif [ -f "DeepMindcheck/manage.py" ]; then
    echo "Found manage.py in DeepMindcheck subdirectory"
    cd DeepMindcheck
    python manage.py collectstatic --noinput
    python manage.py migrate
    gunicorn deepmindcheck.wsgi:application
else
    echo "Cannot find manage.py!"
    find . -name "manage.py"
    exit 1
fi
