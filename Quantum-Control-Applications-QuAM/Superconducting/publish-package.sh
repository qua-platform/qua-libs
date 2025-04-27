#! /bin/bash

#rm -rf dist
#poetry build

export CODEARTIFACT_AUTH_TOKEN=`aws codeartifact get-authorization-token --domain quantum-machines --domain-owner 439440158105 --region us-east-1 --query authorizationToken --output text`

export TWINE_REPOSITORY_URL=https://aws:$CODEARTIFACT_AUTH_TOKEN@quantum-machines-439440158105.d.codeartifact.us-east-1.amazonaws.com/pypi/qm-pypi

#pip install twine
twine upload --verbose --repository codeartifact ./dist/quam*.tar.gz ./dist/quam*.whl
