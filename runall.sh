#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
	bash run.sh $line
done < $1
