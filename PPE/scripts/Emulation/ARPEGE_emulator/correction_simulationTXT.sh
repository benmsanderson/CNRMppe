#!/bin/bash
#
. ~/.bashrc
shopt -s expand_aliases
#
#set -vx
set -u
#
ulimit -s unlimited
export OMP_STACKSIZE=128m
export OMP_NUM_THREADS=4
#
#############################################################################
# Script to find which simulations failed and did not give outputs in 
# /scratch/globc/dcom/ARPEGE6_TUNE. Then save the rows number in a file called  
# 'missing_lines_simulations.txt'
#############################################################################
#
#############################################################################
# 	Check the existing files in /scratch/globc/dcom/ARPEGE6_TUNE/
#############################################################################
#
list_missing=( )
i=0
#
#.......... In the files 0*0*
#
    nb1=1
    while [ $nb1 -lt 10 ]
    do
#
	nb2=1
	while [ $nb2 -lt 10 ]
	do
	    file=/scratch/globc/dcom/ARPEGE6_TUNE/PRE623TUN0${nb1}0${nb2}PL.nc
#
	    if [ -f "$file" ]
	    then
#
                nb2=$(($nb2 + 1))
#
	    else
#
		let "missing=($nb1-1)*10+$nb2"
		list_missing[$i]=$missing 
		i=$(($i + 1))
		nb2=$(($nb2 + 1))
#
	    fi        
#
	done
#
	    nb1=$(($nb1 + 1))
    done
#
#
#
#.......... In the files **0*
#
    nb1=10
    while [ $nb1 -lt 21 ]
    do
#
        nb2=1
        while [ $nb2 -lt 10 ]
        do
            file=/scratch/globc/dcom/ARPEGE6_TUNE/PRE623TUN${nb1}0${nb2}PL.nc
#
            if [ -f "$file" ]
            then
#
                nb2=$(($nb2 + 1))
#
            else
#
                let "missing=($nb1-1)*10+$nb2"
                list_missing[$i]=$missing
                i=$(($i + 1))
                nb2=$(($nb2 + 1))
#
            fi
#
        done
#
            nb1=$(($nb1 + 1))
    done
#
#
#
#.......... In the files **10
#
    nb1=10
    while [ $nb1 -lt 21 ]
    do
#
        nb2=10
        file=/scratch/globc/dcom/ARPEGE6_TUNE/PRE623TUN${nb1}${nb2}PL.nc
#
        if [ -f "$file" ]
        then
#
            nb2=10
#
        else
#
            let "missing=($nb1-1)*10+$nb2"
            list_missing[$i]=$missing
            i=$(($i + 1))
            nb2=$(($nb2 + 1))
#
        fi
#
            nb1=$(($nb1 + 1))
    done
#
#.......... In the files 0*10
#
    nb1=1
    while [ $nb1 -lt 10 ]
    do
#
        nb2=10
        file=/scratch/globc/dcom/ARPEGE6_TUNE/PRE623TUN0${nb1}${nb2}PL.nc
#
        if [ -f "$file" ]
        then
#
            nb2=10
#
        else
#
            let "missing=($nb1-1)*10+$nb2"
            list_missing[$i]=$missing
            i=$(($i + 1))
            nb2=$(($nb2 + 1))
#
        fi
#
            nb1=$(($nb1 + 1))
    done
#
# Print the list of missing numbers :
 for j in ${list_missing[@]}; do echo $j; done
#
#
################################################################################
# 	    Save corresponding rows in the 'Simulations.txt' file
################################################################################
#
#
 printf "%s\n" "${list_missing[*]}" > missing_lines_simulations.txt
#




