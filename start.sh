#!/bin/bash
# Check if we're already in the right directory
if [ -f "manage.py" ]; then
    echo "Found manage.py in current directory"
    python manage.py collectstatic --noinput
    python manage.py migrate
    gunicorn deepmindcheck.wsgi:application
elif [ -f "DeepMindCheck/manage.py" ]; then
    echo "Found manage.py in DeepMindCheck subdirectory"
    cd DeepMindCheck
    python manage.py collectstatic --noinput
    python manage.py migrate
    gunicorn deepmindcheck.wsgi:application
else
    echo "Cannot find manage.py!"
    find . -name "manage.py"
    exit 1
fi
