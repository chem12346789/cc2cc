/bin/top -u chenzihao -o %MEM -bn1 -c -w250 | grep "multiprocessing" | awk '{print $1}' | xargs -I {} sh -c 'echo "Killing process {}"; kill -9 {}'
/bin/top -u chenzihao -o %MEM -bn1 -c -w250 | grep "pickler=torch" | awk '{print $1}' | xargs -I {} sh -c 'echo "Killing process {}"; kill -9 {}'
# find file in /tmp owned by chenzihao and older than 20 day, and delete it. Note that the we may not have permission to show all the files in /tmp, so we need to use find command with -user option to filter the files owned by chenzihao.
find /tmp -user chenzihao -type f -mtime +20 -exec rm -f {} \;
