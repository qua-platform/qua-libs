#!/bin/bash

FILES=$(find -type f -name "*.md")
OUTPUT_FILE=readme.md

echo "# Examples index" > ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}
for i in $FILES; do
    DIRNAME=$(dirname $i)
    if [ ${DIRNAME} != "." ]; then
        echo "- [$(echo ${DIRNAME} | cut -c 3- )]($i)" >> ${OUTPUT_FILE}
    fi
done