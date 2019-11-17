#!/bin/bash

path_jenkins=`grep path_jenkins inputs_deploy.txt | sed -e 's/\(.*\)=\(.*\)$/\2/' \
              | tr -d [:space:]`
deployment_branch=`grep branch inputs_deploy.txt | sed -e 's/\(.*\)=\(.*\)$/\2/' \
                  | tr -d [:space:]`

# Add git color auto
git config --global color.ui auto

# Check input number
if [ $# -gt 1 ]; then
    echo " "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    echo "ERROR number of input parameters should be 0 to 1"
    echo "to see options accepted, type:"
    echo " "
    echo "./deployment -h"
    echo " "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    exit 1
fi

dry_run_input=${1:-no_dry_run}

case $dry_run_input in
--dry-run)
    echo " "
    echo "WARNING: DRY-RUN"
    echo " "
    ;;
no_dry_run)
    echo " "
    echo "DEPLOYMENT"
    echo "----------"
    echo " "
    ;;
-h)
    echo " "
    echo "Usage: ./deployment.sh [-h] [--dry-run]"
    echo " "
    echo "Deployment script for experimental diagnostic treatments"
    echo "after shot"
    echo " "
    echo "optional arguments:"
    echo "-h          show this help message and exit"
    echo "--dry-run   DRY RUN, run git checks but does not deploy"
    echo " "
    exit 0
    ;;
*)
    echo " "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    echo "ERROR: to see options accepted, type:"
    echo " "
    echo "./deployment -h"
    echo " "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    exit 1
    ;;
esac

echo "General information"
echo "-------------------"
echo "whoami:"
whoami
echo "hostname:"
hostname
echo "user linux id:"
id
echo "Current folder:"
pwd
echo "path_jenkins:"
echo $path_jenkins
echo "deployment_branch:"
echo $deployment_branch
#echo "dry_run_input:"
#echo $dry_run_input

echo " "
echo "Git update"
echo "----------"
branch_name=`git rev-parse --abbrev-ref HEAD`
echo "branch_name:"
echo $branch_name
if [ $branch_name != $deployment_branch ]; then
   echo " "
   echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
   echo " "
   echo "ERROR: if you want to deploy please go to $deployment_branch branch"
   echo " "
   echo "Type the command:"
   echo " "
   echo "git checkout $deployment_branch"
   echo " "
   echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
   echo " "
   exit 1
fi
echo " "
echo "git diff test:"
git diff --exit-code --quiet
ret_diff=$?
git diff --cached --exit-code --quiet
ret_diff_cached=$?
if [ $ret_diff -ne 0 ] || [ $ret_diff_cached -ne 0 ]; then
   echo " "
   echo " "
   echo "WARNING: There are not commited changes"
   echo "- IF you WANT TO DEPLOY CHANGES"
   echo "  hit [n] (no) and explanations"
   echo "  to commit changes will appear"
   echo "- IF you want to deploy WITHOUT"
   echo "  changes type [Y] (yes)"
   echo " "
   read -p "[Y/n] ?" -n 1 -r
   echo " "
   if [[ ! $REPLY =~ Y ]]
   then
       echo " "
       echo " "
       echo "To commit the changes type the command:"
       echo "--------------------------------------"
       echo " "
       echo "git commit -am \"YOUR COMMIT MESSAGE\""
       echo " "
       #echo "BASH_SOURCE = $BASH_SOURCE"
       #echo "Zero        = $0"
       [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
   fi
else
   echo "Test passed"
fi
echo " "
echo "git status:"
git status
echo " "
echo "Check untracked files:"
untracked_files=`git ls-files --exclude-standard --others`
if [ -z "$untracked_files" ]; then
    echo "Not untracked files"
else
   echo " "
   echo " "
   echo "WARNING: There are untracked files"
   echo "- IF you WANT TO DEPLOY UNTRACKED FILES"
   echo "  hit [n] (no) and explanations to"
   echo "  track and commit new files will appear"
   echo "- IF you want to deploy WITHOUT"
   echo "  untracked files type [Y] (yes)"
   echo " "
   read -p "[Y/n] ?" -n 1 -r
   echo " "
   if [[ ! $REPLY =~ Y ]]
   then
       echo " "
       echo " "
       echo "To add and commit the new files type the commands:"
       echo "-------------------------------------------------"
       echo " "
       echo "git add NEW_FILE_NAME"
       echo "git commit -m \"YOUR COMMIT MESSAGE\""
       echo " "
       #echo "BASH_SOURCE = $BASH_SOURCE"
       #echo "Zero        = $0"
       [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
   fi
fi
echo " "
echo "Test if pull needed:"
echo " "
echo "git fetch first:"
git fetch
echo " "
echo "Check different commit ID"
#UPSTREAM=${1:-'@{u}'}
LOCAL=$(git rev-parse '@{0}')
echo "LOCAL = $LOCAL"
REMOTE=$(git rev-parse '@{u}')
echo "REMOTE = $REMOTE"
BASE=$(git merge-base '@{0}' '@{u}')
echo "BASE = $BASE"
echo " "
if [ $LOCAL = $REMOTE ]; then
    echo "Up-to-date"
#elif [ $LOCAL = $BASE ]; then
#    echo "Need to pull"
elif [ $REMOTE = $BASE ]; then
    echo "Need to push"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    echo "ERROR: if you want to deploy please pull changes"
    echo " "
    echo "TYPE THE COMMAND:"
    echo " "
    echo "git pull --rebase"
    echo " "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " "
    exit 1
fi
echo " "
echo "git remote show origin:"
git remote show origin
#echo " "
#echo "Files in folder:"
#ls -trlh
echo " "
echo "git show:"
git show -s
echo " "
echo "git push origin master:"
git push origin $deployment_branch

echo " "
echo "Deployment of treatment"
echo "-----------------------"
if [ $dry_run_input = no_dry_run ]; then
    echo " "
    echo "No dry run, deployment"
    echo " "
    curl -v -X POST http://S-CAD-IRFM-TRAIT:e3c97938a60921b09782a5c99d2f7e40@pyxis.intra.cea.fr:8080/${path_jenkins}/build?token=DEPLOY
    ret=$?
    if [ $ret -ne 0 ]; then
        echo ' '
        echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        echo ERROR if you need help with this error please
        echo contact: Jorge Morales: jorge.morales2@cea.fr
        echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        exit 1
    else
        echo ' '
        echo 'Deployment launched'
        echo ' '
    fi
else
    echo " "
    echo "WARNING: DRY-RUN"
    echo " "
fi
