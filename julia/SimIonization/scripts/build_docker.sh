#!/bin/bash
# Build Docker image

if [[ $1 == "" ]]; then
    echo You need to specify a build directory
    exit 1
fi

echo working directory is $(pwd)
echo directory to build is $1

git_branch=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
git_repo=$(basename -s .git `git config --get remote.origin.url`)
git_repo=${git_repo,,}
echo git_repo: ${git_repo}
echo git_branch: ${git_branch}

docker build -t ynashed/${git_repo}-${git_branch}:latest $1